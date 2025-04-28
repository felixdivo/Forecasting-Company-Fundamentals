from typing import Iterable, TypeVar

from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.likelihood_models import QuantileRegression
from darts.timeseries import TimeSeries
from chronos import BaseChronosPipeline
import torch
import pandas as pd


TimeSeriesCollection = TypeVar("TimeSeriesCollection", Iterable[TimeSeries], TimeSeries)


class ChronosDartsWrapper(GlobalForecastingModel):
    """This class wraps the Chronos model from Huggingface's Transformers into a Darts ForecastingModel.

    Args:
        model_name: Name of the loaded model.
        device: Possilbe devices for the model to run on.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device: str = "cpu",
    ):
        super().__init__(add_encoders=None)
        self.model_name = model_name
        self.device = device
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16
        )
        self.regression = QuantileRegression(quantiles=[i * 0.1 for i in range(1, 10)])

    def fit(self, series: TimeSeries) -> None:
        """
        Darts' fit method. Chronos does not require explicit training for pretrained models.
        """

    def predict(
        self,
        n: int,
        series: TimeSeriesCollection,
        num_samples: int,
        verbose: bool = False,
        show_warnings: bool = False,
        **kwargs,
    ) -> TimeSeriesCollection:
        """
        This is the main predict methos. It takes a TimeSeries transforms it into a tensor and
        predicts the quantiles of the next n steps with Chronos. Next a Quantile Regression is used to predict get
        a probabilistic forecast. The forecast is then transformed back into a TimeSeries object.

        Args:
            n (int): Number of steps to predict.
            series (TimeSeries): The input TimeSeries to predict on.
            verbose (bool, optional): Ignored.
            show_warnings (bool, optional): Ignored.
            kwargs (bool, optional): Ignored.

        Returns:
            Union[TimeSeries, List[TimeSeries]]: The predicted TimeSeries or a list of TimeSeries.
        """
        if kwargs:
            print(
                f"Warning: The following arguments are not supported and will be ignored: {kwargs}"
            )

        # Ensure input series is a list of TimeSeries objects
        if isinstance(series, TimeSeries):
            series = [series]

        # Prepare context: decompose multivariate series into separate lists of tensors
        # To run this on a gpu, the tensors must be on not be on the gpu! Data transfer is handled by Chronos
        context = []
        for ts in series:
            for variate_idx in range(ts.n_components):
                context.append(
                    torch.from_numpy(
                        ts.univariate_component(variate_idx).values().squeeze()
                    )
                )

        # Generate predictions using ChronosPipeline
        # This returns (n_timesteps, quantiles, batch_size), mean
        quantiles, mean = self.pipeline.predict_quantiles(
            context=context,
            prediction_length=n,
        )

        # add the extra dimension n_components for the Darts Regressor
        quantiles = quantiles.unsqueeze(2)

        # The darts regressor takes an input with shape (n_samples, n_timesteps, n_components, n_quantiles)
        # It returns a tensor of shape (n_samples, n_timesteps, n_components)
        # Since components are modeled as samples, n_components is 1
        samples = []
        for _ in range(num_samples):
            forecast = self.regression.sample(quantiles).squeeze()
            samples.append(forecast)

        forecast = torch.stack(samples, dim=0)

        # transform back to time series format
        # TimeSeries expects an input of shape (n_timesteps, n_components)
        # If the input was a univariate series, we need to add the extra dimension
        if forecast.dim() == 2:
            forecast = forecast.unsqueeze(0)

        forecast = forecast.permute(0, 2, 1).cpu().numpy()
        forecast = forecast.reshape(n, -1, num_samples)

        # Create time index for predictions
        if isinstance(ts.time_index, pd.DatetimeIndex):
            time_index = pd.date_range(
                start=ts.time_index[-1] + ts.freq, periods=n, freq=ts.freq
            )
        elif isinstance(ts.time_index, pd.RangeIndex):
            time_index = pd.RangeIndex(
                start=ts.time_index[-1] + 1, stop=ts.time_index[-1] + n + 1
            )

        predicted_series_list = []
        for idx, ts in enumerate(series):
            # Extract predictions for this series variates
            num_variates = ts.width
            single_prediction = forecast[
                :, idx * num_variates : (idx + 1) * num_variates, :
            ]

            # Convert predictions to TimeSeries
            # TimeSeries expects an input of shape (n_timesteps, n_components)
            predicted_series = TimeSeries.from_times_and_values(
                time_index, single_prediction
            )
            predicted_series_list.append(predicted_series)

        # Return a single series if input was single, else a list of predictions
        return (
            predicted_series_list
            if len(predicted_series_list) > 1
            else predicted_series_list[0]
        )

    """
    This are Darts ForecastingModule Methods, this was generated by ChatGPT
    NOTE: only changed multivariate to True
    """

    @staticmethod
    def supports_multivariate() -> bool:
        """
        Specifies if the model supports multivariate series.
        """
        return True

    @staticmethod
    def supports_transferrable_series_prediction() -> bool:
        """
        Specifies if the model supports transferring predictions across series.
        """
        return True

    @staticmethod
    def _model_encoder_settings():
        """
        Returns model-specific encoder settings. Not applicable for Chronos.
        """
        return None

    @staticmethod
    def extreme_lags() -> tuple:
        """
        Returns the minimum and maximum lag supported by the model.
        """
        return 0, 0  # Chronos uses full context for predictions.
