"""Based om https://github.com/PaulRabich/darts/tree/xlstm-mixer."""

from typing import Literal

import torch
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import AvgPool1d, Linear, Module, Parameter, Sequential
from xlstm import (
    mLSTMBlockConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

from darts.models.components.layer_norm_variants import RINorm
from darts.models.forecasting.pl_forecasting_module import (
    PLPastCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel


class moving_avg(Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class _xLSTMMixer(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        xlstm_embedding_dim: int,
        num_mem_tokens: int,
        num_tokens_per_variate: int,
        xlstm_dropout: float,
        xlstm_conv1d_kernel_size: int,
        xlstm_num_heads: int,
        xlstm_num_blocks: int,
        use_mlstm: bool,
        use_revin: bool,
        packing: int,
        backbone: Literal["nlinear", "dlinear"],
        nr_params,
        **kwargs,
    ) -> None:
        """PyTorch module implementing the xLSTMMixer architecture.

        Parameters
        ----------
        input_dim : int
            Numer of variables in the input time series
        output_dim : int
            Number of variables in the output time series
        xlstm_embedding_dim : int
            _description_
        num_mem_tokens : int
            _description_
        num_tokens_per_variate : int
            _description_
        xlstm_dropout : float
            _description_
        xlstm_conv1d_kernel_size : int
            _description_
        xlstm_num_heads : int
            _description_
        xlstm_num_blocks : int
            _description_
        use_mlstm : bool
            _description_
        use_revin : bool
            Wether to apply reversible instance normalization. Note RINorm is applied to all variates of the input.
        packing : int
            _description_
        backbone : Literal[&quot;nlinear&quot;, &quot;dlinear&quot;]
            _description_
        nr_params : _type_
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.xlstm_embedding_dim = xlstm_embedding_dim
        self.use_mlstm = use_mlstm
        self.use_revin = use_revin
        self.packing = packing
        self.nr_params = nr_params

        self.mem_tokens = (
            Parameter(torch.randn(num_mem_tokens, xlstm_embedding_dim) * 0.01)
            if num_mem_tokens > 0
            else None
        )

        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=xlstm_num_heads, conv1d_kernel_size=xlstm_conv1d_kernel_size
            )
        )

        self.mlp_in = Sequential(
            Linear(
                self.input_chunk_length,
                self.output_chunk_length * num_tokens_per_variate,
            ),
        )

        self.mlp_in_trend = Sequential(
            Linear(
                self.input_chunk_length,
                self.output_chunk_length * num_tokens_per_variate,
            ),
        )

        self.pre_encoding = Linear(self.output_chunk_length, xlstm_embedding_dim)

        self.xlstm = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=(mLSTMBlockConfig() if use_mlstm else None),
                slstm_block=slstm_config,
                num_blocks=xlstm_num_blocks,
                embedding_dim=xlstm_embedding_dim
                * self.packing,  # @Marice, packing rauslöschen?? Debugging? Wenn nicht default Wert ausdenken
                add_post_blocks_norm=True,
                dropout=xlstm_dropout,
                bias=True,
                slstm_at=([] if self.use_mlstm else "all"),
                context_length=self.input_dim * num_tokens_per_variate + num_mem_tokens,
            )
        )

        self.fc = Linear(
            self.xlstm_embedding_dim * 2,
            self.nr_params * self.output_chunk_length,  # removed self.output_dim
        )
        self.norm = RINorm(self.input_dim, affine=self.use_revin)
        self.decomposition = series_decomp(25)
        self.seq_var_2_var_seq = Rearrange("batch seq var -> batch var seq")
        self.var_seq_2_seq_var = Rearrange("batch var seq -> batch seq var")

        self.Linear = Linear(self.input_chunk_length, self.output_chunk_length)
        self.backbone = backbone

    @io_processor
    def forward(self, x_in) -> Tensor:
        # x then has shape [Batch, Input length, Target Variates + Covariates]
        x, _ = x_in

        # Normalize covariates too. Norm needs shape (b, seq, var)
        # x[:, :, self.n_targets :] = self.rin(x[:, :, self.n_targets :])

        if self.backbone == "nlinear":
            # NLinear
            # x: [Batch, Input length, Channel]
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x_pre_forecast = x + seq_last
            x_pre_forecast = self.seq_var_2_var_seq(x_pre_forecast)

        elif self.backbone == "dlinear":
            seasonal_init, trend_init = self.decomposition(x)

            seasonal_init = self.seq_var_2_var_seq(seasonal_init)
            trend_init = self.seq_var_2_var_seq(trend_init)

            seasonal_init = self.mlp_in(seasonal_init)
            trend_init = self.mlp_in_trend(trend_init)
            x_pre_forecast = seasonal_init + trend_init
        else:
            raise ValueError("Unknown backbone")

        x = self.pre_encoding(x_pre_forecast)

        if self.packing > 1:
            var = x.shape[1]
            assert var % self.packing == 0, (
                "The number of variables must be divisible by n"
            )

            # Pack variables into sequence
            x = rearrange(x, "b (n var) seq -> b var (seq n)", n=self.packing)

        if self.mem_tokens is not None:
            m: Tensor = repeat(self.mem_tokens, "m d -> b m d", b=x.shape[0])
            x, mem_ps = pack([m, x], "b * d")

        # Backward view
        dim = -1
        x_reversed = torch.flip(x, [dim])
        x_bwd = self.xlstm(x_reversed)

        # Forward view
        x = self.xlstm(x)

        # Combined the two views
        x = torch.cat((x, x_bwd), dim=dim)

        if self.mem_tokens is not None:
            # Remove the memory tokens
            m, x = unpack(x, mem_ps, "b * d")

        # We only want to only use the target variates for the prediction
        x = x[:, :self.n_targets, ...]

        if self.packing > 1:
            x = rearrange(x, "b var (seq n) -> b (var n) seq", n=self.packing)

        # Perform view mixing
        y = self.fc(x)

        # Rearrange for darts output
        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)

        return y


class xLSTMMixer(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        xlstm_embedding_dim: int = 256,
        num_mem_tokens: int = 1,
        num_tokens_per_variate: int = 1,
        xlstm_dropout: float = 0.2,
        xlstm_conv1d_kernel_size: int = 4,
        xlstm_num_heads: int = 8,
        xlstm_num_blocks: int = 2,
        use_mlstm: bool = False,
        use_reversible_instance_norm: bool = False,
        packing: int = 1,
        backbone: Literal["nlinear", "dlinear"] = "nlinear",
        **kwargs,
    ):
        """xLSTMMixer model

        Parameters
        ----------
        input_chunk_length : int
            Length of the input time series.
        output_chunk_length : int
            Length of the output time series.
        output_chunk_shift : int, optional
            _description_, by default 0
        xlstm_embedding_dim : int, optional
            _description_, by default 256
        num_mem_tokens : int, optional
            _description_, by default 12
        num_tokens_per_variate : int, optional
            _description_, by default 1
        xlstm_dropout : float, optional
            _description_, by default 0
        xlstm_conv1d_kernel_size : int, optional
            _description_, by default 2
        xlstm_num_heads : int, optional
            _description_, by default 2
        xlstm_num_blocks : int, optional
            _description_, by default 4
        use_mlstm : bool, optional
            _description_, by default False
        use_reversible_instance_norm : bool, optional
            _description_, by default True
        packing : int, optional
            _description_, by default 1
        backbone : Literal[&quot;nlinear&quot;], optional
            _description_, by default "nlinear"
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.xlstm_embedding_dim = xlstm_embedding_dim
        self.num_mem_tokens = num_mem_tokens
        self.num_tokens_per_variate = num_tokens_per_variate
        self.xlstm_dropout = xlstm_dropout
        self.xlstm_conv1d_kernel_size = xlstm_conv1d_kernel_size
        self.xlstm_num_heads = xlstm_num_heads
        self.xlstm_num_blocks = xlstm_num_blocks
        self.use_mlstm = use_mlstm
        self.use_reversible_instance_norm = use_reversible_instance_norm
        self.packing = packing
        self.backbone = backbone

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        input_dim = train_sample[0].shape[-1]
        output_dim = train_sample[-1].shape[1]

        self.use_reversible_instance_norm = False

        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _xLSTMMixer(
            input_dim=input_dim,
            output_dim=output_dim,
            xlstm_embedding_dim=self.xlstm_embedding_dim,
            num_mem_tokens=self.num_mem_tokens,
            num_tokens_per_variate=self.num_tokens_per_variate,
            xlstm_dropout=self.xlstm_dropout,
            xlstm_conv1d_kernel_size=self.xlstm_conv1d_kernel_size,
            xlstm_num_heads=self.xlstm_num_heads,
            xlstm_num_blocks=self.xlstm_num_blocks,
            use_mlstm=self.use_mlstm,
            use_revin=self.use_reversible_instance_norm,
            packing=self.packing,
            backbone=self.backbone,
            nr_params=nr_params,
            **self.pl_module_params,
        )
