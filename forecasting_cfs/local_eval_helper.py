"""This heloer file is required since we cannot perform multiprocessing in a Jupyter Notebook directly."""

from typing import ContextManager, Optional
from itertools import repeat, starmap
from functools import partial
import multiprocessing

from darts import TimeSeries, concatenate
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)


def _local_fit_predict(
    model: ForecastingModel,
    ts_in_single: TimeSeries,
    past_covariates_single: Optional[TimeSeries],
    num_output_steps: int,
    normalize_local_models: bool,
    num_samples: int = 1,
) -> TimeSeries:
    if normalize_local_models:
        mean = ts_in_single.univariate_values().mean().item()
        ts_in_single = ts_in_single - mean
    else:
        mean = 0

    model = model.untrained_model()
    try:
        model.fit(ts_in_single, past_covariates=past_covariates_single)
    except TypeError:
        # model.supports_past_covariates is not reliable
        model.fit(ts_in_single)

    return model.predict(num_output_steps, num_samples=num_samples) + mean


# We somehow can't subclass Pool directly
class MultivariateLocalModelEvaluator(ContextManager):
    def __init__(
        self, *, parallelize: bool = True, paralell_chunksize: int = 25, **kwargs
    ):
        self.parallelize = parallelize
        self.paralell_chunksize = paralell_chunksize
        self.kwargs = kwargs

        self.pool = None

        super().__init__()

        # We init later, since we can't pickle the pool

    def _lazy_init(self):
        """May be called multiple times, but only initializes once."""
        if not hasattr(self, "mapper"):
            if self.parallelize:
                mp_context = multiprocessing.get_context("spawn")  # Needed for CUDA
                self.pool = mp_context.Pool(**self.kwargs)
                self.pool.__enter__()
                self.mapper = partial(
                    self.pool.starmap, chunksize=self.paralell_chunksize
                )
            else:
                # the other defaults are fine in this case
                self.mapper = starmap

    def __enter__(self):
        if self.pool is not None:
            self.pool.__enter__()
        return self

    def __exit__(self, *args):
        if self.pool is not None:
            return self.pool.__exit__(*args)
        else:
            return False

    def evaluate(
        self,
        *,
        model: ForecastingModel,
        ts_in: TimeSeries,
        past_covariates: TimeSeries,
        num_output_steps: int,
        normalize_local_models: bool,
        num_samples: int = 1,
    ) -> TimeSeries:
        if isinstance(model, GlobalForecastingModel):
            raise ValueError("Model must be a local model")

        ts_in_univariates = (
            ts_in.univariate_component(component)
            for component in range(ts_in.n_components)
        )

        # Rare, but conceivable for univariate models
        if model.supports_past_covariates:
            past_covariates_univariate = (
                past_covariates.univariate_component(component)
                for component in range(past_covariates.n_components)
            )
        else:
            past_covariates_univariate = repeat(None)

        self._lazy_init()  # can be called multiple times
        individual_predictions = self.mapper(
            _local_fit_predict,
            zip(
                repeat(model),
                ts_in_univariates,
                past_covariates_univariate,
                repeat(num_output_steps),
                repeat(normalize_local_models),
                repeat(num_samples),
            ),
        )

        return concatenate(
            list(individual_predictions),
            axis="component",
            ignore_static_covariates=True,
        )
