from typing import Any, ContextManager, Optional, Generator, Sequence
from pathlib import Path
from dataclasses import dataclass
from traceback import format_exception

# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing.context import SpawnContext
# import torch.multiprocessing as mp
#
# class PytorchContext(SpawnContext):
#     def SimpleQueue(self):
#         return mp.SimpleQueue()
#
#     def Queue(self, maxsize=0):
#         return mp.Queue(maxsize, ctx=self.get_context())
#
# mp_context = PytorchContext()


import numpy as np
import torch
from torch import Tensor
from torchmetrics import MetricCollection, MultioutputWrapper, Metric
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    RelativeSquaredError,
)

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.models import Prophet

import wandb
from lightning.pytorch.loggers import WandbLogger
from rtpt import RTPT
from rich.progress import track

from acatis_data.darts import TimeSeriesContainer
from acatis_data import KEY_FEATURE_NAMES

from .local_eval_helper import MultivariateLocalModelEvaluator


def is_global(model: ForecastingModel) -> bool:
    return isinstance(model, GlobalForecastingModel)


def to_torch(tss: Sequence[TimeSeries]) -> Tensor:
    """Returns `(n_ts, D_TEST, num_target_features)`."""
    as_numpy = np.stack([ts.values() for ts in tss], axis=0)  # Disregards samples
    return torch.from_numpy(as_numpy)


def to_torch_avg_all(tss: Sequence[TimeSeries]) -> Tensor:
    return to_torch(tss).flatten()


def to_torch_avg_feature(tss: Sequence[TimeSeries]) -> Tensor:
    # The feature dimension is already last (num_target_features)
    return to_torch(tss).flatten(0, 1)


def to_torch_avg_lookahead(tss: Sequence[TimeSeries]) -> Tensor:
    # Move the lookahead dimension to the end (D_TEST)
    return to_torch(tss).transpose(-1, -2).flatten(0, 1)


try:
    _human_eval_companies = set(np.loadtxt("human_eval_companies.csv", delimiter=","))
except FileNotFoundError:
    _human_eval_companies = set(
        np.loadtxt(
            "acatis-applications/notebooks/human_eval_companies.csv", delimiter=","
        )
    )


def to_torch_avg_1y_ahead_revenue(
    tss: Sequence[TimeSeries], meta_data: Sequence[dict[str, any]]
) -> Tensor:
    as_torch = to_torch(
        [
            ts
            for ts, meta in zip(tss, meta_data)
            if meta["companyid"] in _human_eval_companies
        ]
    )
    assert as_torch.shape[1] == 4
    assert as_torch.shape[2] == len(KEY_FEATURE_NAMES)
    feature_index = KEY_FEATURE_NAMES.index("Total Revenues")
    return as_torch[:, -1, feature_index]


def wrap_per_feature(metric: Metric, num_targets: int) -> Metric:
    return MultioutputWrapper(metric, num_targets, remove_nans=False)


def wrap_per_lookahead(metric: Metric, lookahead_steps: int) -> Metric:
    return MultioutputWrapper(metric, lookahead_steps, remove_nans=False)


@dataclass(frozen=True, kw_only=True)
class ForecastingResult:
    ts_past: TimeSeries
    ts_ground_truth: TimeSeries
    ts_forecast: TimeSeries
    meta_data: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class ModelEvalResult:
    name: str
    metrics: dict[str, float] | None
    test_forecasts: list[ForecastingResult] | None


class ModelEvaluator(ContextManager):
    def __init__(
        self,
        *,
        train_ts: TimeSeriesContainer,
        val_ts: TimeSeriesContainer,
        test_ts: TimeSeriesContainer,
        input_chunk_length: int,
        output_chunk_length: int,
        checkpoint_path: Path,
        normalize_local_models: tuple[ForecastingModel] = (Prophet,),
        num_samples: int = 1,
        use_logger: bool = True,
        group_name: str = "debugging",
        job_type: Optional[str] = None,
        evaluate_first_n_companies: int = 1_000_000_000,  # all of them
        return_first_n_companies: int = 1_000_000_000,  # all of them
        num_processes_local_models: int = 1,  # 1 disables multiprocessing
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.normalize_local_models = normalize_local_models
        self.num_samples = num_samples
        self.use_logger = use_logger
        self.group_name = group_name
        self.job_type = job_type

        self.evaluate_first_n_companies = evaluate_first_n_companies
        self.return_first_n_companies = return_first_n_companies

        self.train_ts = train_ts
        self.val_ts = val_ts
        self.test_ts = test_ts

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        assert (
            self.input_chunk_length > 0 and self.output_chunk_length > 0
        ), "input and output chunk length must be positive"
        assert (
            self.input_chunk_length + self.output_chunk_length
            <= self.train_ts.targets[0].n_timesteps
        ), "input and output chunk length must not be longer than the time series"

        # test_ts.targets, .covariates are lists of TimeSeries, so we need to iterate over each entry individually
        # This takes a while to compute, so we only do it once
        print("Rearranging the test time series into input & output chunks...")
        self.test_targets_in, self.test_targets_out = zip(
            *(ts.split_before(self.input_chunk_length) for ts in test_ts.targets)
        )
        self.test_covariates_in, _ = zip(
            *(ts.split_before(self.input_chunk_length) for ts in test_ts.covariates)
        )
        print("Done.")

        self._init_metrics()

        self.local_evaluator = MultivariateLocalModelEvaluator(
            parallelize=num_processes_local_models != 1,
            processes=num_processes_local_models,
        )

    def _init_metrics(self) -> None:
        self.metrics_avg_all = MetricCollection(
            {
                "MAE (avg)": MeanAbsoluteError(),
                "MSE (avg)": MeanSquaredError(),
                "RMSE (avg)": MeanSquaredError(squared=False),
                "R2 (avg)": R2Score(),
                "MAPE (avg)": MeanAbsolutePercentageError(),
                "SMAPE (avg)": SymmetricMeanAbsolutePercentageError(),
                "RSE (avg)": RelativeSquaredError(),
            }
        )

        n_components = self.train_ts.targets[0].n_components
        self.metrics_avg_feature = MetricCollection(
            {
                "MAE (avg per feature)": wrap_per_feature(
                    MeanAbsoluteError(), n_components
                ),
                "MSE (avg per feature)": wrap_per_feature(
                    MeanSquaredError(), n_components
                ),
                "RMSE (avg per feature)": wrap_per_feature(
                    MeanSquaredError(squared=False), n_components
                ),
                "R2 (avg per feature)": wrap_per_feature(R2Score(), n_components),
                "MAPE (avg per feature)": wrap_per_feature(
                    MeanAbsolutePercentageError(), n_components
                ),
                "SMAPE (avg per feature)": wrap_per_feature(
                    SymmetricMeanAbsolutePercentageError(), n_components
                ),
                "RSE (avg per feature)": wrap_per_feature(
                    RelativeSquaredError(), n_components
                ),
            }
        )

        self.metrics_per_lookahead = MetricCollection(
            {
                "MAE (avg per lookahead)": wrap_per_lookahead(
                    MeanAbsoluteError(), self.output_chunk_length
                ),
                "MSE (avg per lookahead)": wrap_per_lookahead(
                    MeanSquaredError(), self.output_chunk_length
                ),
                "RMSE (avg per lookahead)": wrap_per_lookahead(
                    MeanSquaredError(squared=False), self.output_chunk_length
                ),
                "R2 (avg per lookahead)": wrap_per_lookahead(
                    R2Score(), self.output_chunk_length
                ),
                "MAPE (avg per lookahead)": wrap_per_lookahead(
                    MeanAbsolutePercentageError(), self.output_chunk_length
                ),
                "SMAPE (avg per lookahead)": wrap_per_lookahead(
                    SymmetricMeanAbsolutePercentageError(), self.output_chunk_length
                ),
                "RSE (avg per lookahead)": wrap_per_lookahead(
                    RelativeSquaredError(), self.output_chunk_length
                ),
            }
        )

        self.metrics_1y_ahead_revenue = MetricCollection(
            {
                "MAE (avg as human eval)": MeanAbsoluteError(),
                "MSE (avg as human eval)": MeanSquaredError(),
                "RMSE (avg as human eval)": MeanSquaredError(squared=False),
                "R2 (avg as human eval)": R2Score(),
                "MAPE (avg as human eval)": MeanAbsolutePercentageError(),
                "SMAPE (avg as human eval)": SymmetricMeanAbsolutePercentageError(),
                "RSE (avg as human eval)": RelativeSquaredError(),
            }
        )

    def eval_in_parallel(
        self,
        models: Sequence[ForecastingModel],
        *,
        num_workers: int,
        progress_bar: bool = False,  # We have individual progress bars for each model
    ) -> Generator[ModelEvalResult, None, None]:
        yield from _eval_in_parallel(
            self, models, num_workers=num_workers, progress_bar=progress_bar
        )

    @property
    def has_past_covariates(self) -> bool:
        example = self.train_ts.covariates[0]
        return example is not None and example.n_components > 0

    def __enter__(self):
        self.local_evaluator.__enter__()
        return self

    def __exit__(self, *args):
        return self.local_evaluator.__exit__(*args)

    def _get_logger(self, model_name: str) -> Optional[WandbLogger]:
        if self.use_logger:
            return WandbLogger(
                entity="felix-divo",
                project="ACATIS",
                group=self.group_name,
                job_type=self.job_type,
                name=model_name,
            )
        else:
            return None

    def _train_global_model(
        self, name: str, model: ForecastingModel
    ) -> ForecastingModel:
        checkpoint_path = self.checkpoint_path / f"{name}.ckpt"
        if checkpoint_path.exists():
            print("loading model from checkpoint")
            return model.load(str(checkpoint_path))
        else:
            print("model checkpoint missing, training multi-time series model ...")

            logger: Optional[WandbLogger] = None
            if hasattr(model, "trainer_params"):
                logger = self._get_logger(name)
                model.trainer_params["logger"] = logger

            # if hasattr(model, "fit_from_dataset"):
            #    model.fit_from_dataset(train_ts)

            train_data = dict(series=self.train_ts.targets)
            val_data = dict(val_series=self.val_ts.targets)
            if model.supports_past_covariates and self.has_past_covariates:
                train_data["past_covariates"] = self.train_ts.covariates
                val_data["val_past_covariates"] = self.val_ts.covariates

            try:
                model.fit(**train_data, **val_data)
            except TypeError:
                print("Model does not support validation set tracking")
                model.fit(**train_data)

            if logger is not None:
                logger.experiment.finish()
                wandb.finish()
                model.trainer.logger = None
                model.trainer_params["logger"] = False  # sic.

            try:
                model.save(str(checkpoint_path))
            except Exception as e:
                print(
                    f"Error while saving model checkpoint {checkpoint_path}:\n{e}"
                )
                print("Model will not be saved, but training will continue.")

            return model

    def _eval_single_model(
        self, name: str, model: ForecastingModel
    ) -> (
        None
        | tuple[
            list[TimeSeries] | None,
            list[TimeSeries] | None,
            list[TimeSeries] | None,
            list[dict[str, Any]] | None,
        ]
    ):
        print("-" * 40)
        print(f"Evaluating {name}")

        try:
            if is_global(model):
                model = self._train_global_model(name, model)

                past_covariates_kwargs = (
                    dict(past_covariates=self.test_covariates_in)
                    if model.supports_past_covariates and self.has_past_covariates
                    else {}
                )
                test_forecast = model.predict(
                    self.output_chunk_length,
                    self.test_targets_in,
                    **past_covariates_kwargs,
                    verbose=False,
                    show_warnings=False,
                    num_samples=self.num_samples,
                )
                test_ts_in_to_consider = self.test_targets_in
                test_ts_out_to_consider = self.test_targets_out
            else:
                print(
                    "model checkpoint missing, but training will not be performed now since we train on each series individually"
                )

                # This might take a while, since we need to train the model for each time series too
                test_forecast = []
                test_ts_in_to_consider = self.test_targets_in[
                    : self.evaluate_first_n_companies
                ]
                test_ts_out_to_consider = self.test_targets_out[
                    : self.evaluate_first_n_companies
                ]
                test_covariates_in_to_consider = self.test_covariates_in[
                    : self.evaluate_first_n_companies
                ]
                for ts_in, covariates_in, ts_meta in track(
                    list(
                        zip(
                            test_ts_in_to_consider,
                            test_covariates_in_to_consider,
                            self.test_ts.meta_data,
                        )
                    ),
                    description=f"Training & evaluating {name}",
                ):
                    try:
                        normalize_model = isinstance(model, self.normalize_local_models)

                        # Try if the model can handle multivariate time series
                        try:
                            if (
                                model.supports_past_covariates
                                and self.has_past_covariates
                            ):
                                model.fit(ts_in, past_covariates=covariates_in)
                            else:
                                model.fit(ts_in)
                            test_forecast.append(
                                model.predict(
                                    self.output_chunk_length,
                                    verbose=False,
                                    show_warnings=False,
                                    num_samples=self.num_samples,
                                )
                            )
                            model = model.untrained_model()  # Reset the model!

                            if normalize_model:
                                raise NotImplementedError(
                                    "This model is not supported for multivariate time series"
                                )

                        except ValueError as e:
                            if (
                                "only supports univariate TimeSeries instances"
                                not in str(e)
                            ):
                                # Then this is an error we don't know how to handle!
                                raise e

                            # If not, we need to train & evaluate the model for each component individually
                            test_forecast.append(
                                self.local_evaluator.evaluate(
                                    model=model,
                                    ts_in=ts_in,
                                    past_covariates=covariates_in,
                                    num_output_steps=self.output_chunk_length,
                                    normalize_local_models=normalize_model,
                                    num_samples=self.num_samples,
                                )
                            )

                    except Exception as e:
                        info = "\n".join(format_exception(e))
                        print(
                            f"Error while evaluating {name} @ company ID {ts_meta['companyid']}:\n{info}"
                        )
                        test_forecast.append(None)

            # Possibly only return a subset to not blow up the memory/disk
            return (
                test_ts_in_to_consider[: self.return_first_n_companies],
                test_ts_out_to_consider[: self.return_first_n_companies],
                test_forecast[: self.return_first_n_companies],
                self.test_ts.meta_data[: self.return_first_n_companies],
            )

        except Exception as e:
            info = "\n".join(format_exception(e))
            print(f"Error while evaluating {name}:\n{info}")
            return None


# This needs to be done here because of pickling issues
def _eval_in_parallel(
    evaluator: ModelEvaluator,
    models: dict[str, ForecastingModel],
    *,
    num_workers: int,
    progress_bar: bool,
) -> Generator[ModelEvalResult, None, None]:
    rtpt = RTPT(
        name_initials="FD",
        experiment_name="ForecastBaselines-FitAndEval",
        max_iterations=len(models),
    )
    rtpt.start()

    # mp_context = multiprocessing.get_context("spawn")  # Needed for CUDA
    # mp_context = PytorchContext()
    # with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as pool:
    print("Evaluating models in sequence since multiprocessing is buggy with pytorch")

    for name, single_results in track(
        zip(
            models,
            (
                evaluator._eval_single_model(name, model)
                for name, model in models.items()
            ),
        ),
        description="Fit and evaluate models",
        total=len(models),
        disable=not progress_bar,
        show_speed=True,  # This is too unstable from model to model
    ):
        # single_results = single_results_future.result()
        if single_results is None:
            yield ModelEvalResult(name=name, metrics={}, test_forecasts=[])
            continue

        (
            all_ts_in,
            all_ts_ground_truth,
            all_ts_forecast,
            meta_data,
        ) = single_results

        results = [
            ForecastingResult(
                ts_past=ts_in,
                ts_ground_truth=ts_out,
                ts_forecast=ts_pred,
                meta_data=ts_meta,
            )
            for ts_in, ts_out, ts_pred, ts_meta in zip(
                all_ts_in,
                all_ts_ground_truth,
                all_ts_forecast,
                meta_data,
            )
        ]

        try:
            metrics: dict[str, Any] = {}

            if evaluator.num_samples == 1:
                metrics.update(
                    evaluator.metrics_avg_all(
                        to_torch_avg_all(all_ts_forecast),
                        to_torch_avg_all(all_ts_ground_truth),
                    )
                )
                metrics.update(
                    evaluator.metrics_avg_feature(
                        to_torch_avg_feature(all_ts_forecast),
                        to_torch_avg_feature(all_ts_ground_truth),
                    )
                )
                metrics.update(
                    evaluator.metrics_per_lookahead(
                        to_torch_avg_lookahead(all_ts_forecast),
                        to_torch_avg_lookahead(all_ts_ground_truth),
                    )
                )
                metrics.update(
                    evaluator.metrics_1y_ahead_revenue(
                        to_torch_avg_1y_ahead_revenue(all_ts_forecast, meta_data),
                        to_torch_avg_1y_ahead_revenue(all_ts_ground_truth, meta_data),
                    )
                )

            yield ModelEvalResult(name=name, metrics=metrics, test_forecasts=results)

        except AttributeError as e:
            print(f"Error while evaluating {name}:\n{e}")
            yield ModelEvalResult(name=name, metrics={}, test_forecasts=results)

        rtpt.step(f"Done {name}")
