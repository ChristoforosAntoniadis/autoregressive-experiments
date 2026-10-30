"""
The PyTorch sequence networks: LSTM, GRU, TCN, N-BEATS and DLinear, each predicting next month's
change of one series from the last L months of both series.

They read the same feature rows as every other lag-feature model (ml_forecasting's lag_features:
each series' last L changes, most recent first, standardized per column) and reshape them into a
sequence in time order, shape (L, 2), so features, scaling and recursive forecasts stay the same
for every model; L is the tuned "lags".

- LSTM (Hochreiter and Schmidhuber 1997) and GRU (Cho et al. 2014): stacked recurrent layers,
  the last time step's hidden state through dropout to one linear output
- TCN (Bai, Kolter and Koltun 2018): residual blocks of two dilated causal convolutions each,
  dilation doubling per block; the last time step's channels to one linear output
- N-BEATS, generic (Oreshkin et al. 2020): blocks of fully connected layers, each producing a
  backcast removed from its input and a forecast added to the total
- DLinear (Zeng et al. 2023): a moving-average trend and the remainder, each mapped to the
  forecast by one linear layer

Input: feature rows from ml_forecasting's pipeline, through torch_network.TorchNetwork.

Output: PyTorch modules; nothing is written to disk.
"""

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

SEQUENCE_ARCHITECTURES = ("lstm", "gru", "tcn", "nbeats", "dlinear")
# Both series enter every time step.
SERIES = 2


def to_sequence(rows: torch.Tensor) -> torch.Tensor:
    """
    Reshape lag-feature rows into sequences in time order.

    Parameters
    ----------
    rows : torch.Tensor
        Shape (batch, SERIES x L): the first series' last L changes, most recent first, then the
        second series' the same way (ml_forecasting.lag_features' layout).

    Returns
    -------
    torch.Tensor
        Shape (batch, L, SERIES): the oldest month first, one column per series.
    """
    batch, width = rows.shape
    return rows.reshape(batch, SERIES, width // SERIES).flip(-1).transpose(1, 2)


class Recurrent(nn.Module):
    """
    An LSTM or GRU forecaster.

    Parameters
    ----------
    cell : str
        "lstm" or "gru".
    hidden : int
        Hidden units per recurrent layer.
    layers : int
        Stacked recurrent layers.
    dropout : float
        Dropout between stacked layers and before the output layer.
    bias : bool
        Whether the layers have biases.
    """

    def __init__(self, cell: str, hidden: int, layers: int, dropout: float, bias: bool) -> None:
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        # PyTorch applies its dropout between stacked layers only, so one layer gets none there.
        self.recurrent = recurrent(
            SERIES,
            hidden,
            num_layers=layers,
            bias=bias,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1, bias=bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Forecast from a batch of rows.

        Parameters
        ----------
        rows : torch.Tensor
            Lag-feature rows, shape (batch, SERIES x L).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1).
        """
        output, _ = self.recurrent(to_sequence(rows))
        return self.head(self.dropout(output[:, -1]))


class CausalBlock(nn.Module):
    """
    A TCN residual block: two dilated causal convolutions with ReLU and dropout, plus a skip.

    Parameters
    ----------
    inputs : int
        Input channels.
    channels : int
        Output channels.
    kernel : int
        Convolution kernel size.
    dilation : int
        Gap between the kernel's taps.
    dropout : float
        Dropout after each convolution.
    bias : bool
        Whether the convolutions have biases.

    Notes
    -----
    Causal: each output step sees only its own and earlier steps, because the input is padded
    on the left by (kernel - 1) x dilation and never on the right. The skip is a 1 x 1
    convolution when the channel count changes.
    """

    def __init__(
        self, inputs: int, channels: int, kernel: int, dilation: int, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.first = nn.Conv1d(inputs, channels, kernel, dilation=dilation, bias=bias)
        self.second = nn.Conv1d(channels, channels, kernel, dilation=dilation, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.skip: nn.Module = (
            nn.Conv1d(inputs, channels, 1, bias=bias) if inputs != channels else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : torch.Tensor
            Shape (batch, inputs, L).

        Returns
        -------
        torch.Tensor
            Shape (batch, channels, L).
        """
        hidden = self.dropout(F.relu(self.first(F.pad(values, (self.padding, 0)))))
        hidden = self.dropout(F.relu(self.second(F.pad(hidden, (self.padding, 0)))))
        return F.relu(hidden + self.skip(values))


class TemporalConvolution(nn.Module):
    """
    A TCN forecaster: causal blocks with dilations 1, 2, 4, ..., then one linear output.

    Parameters
    ----------
    hidden : int
        Channels per block.
    layers : int
        Residual blocks.
    kernel : int
        Convolution kernel size.
    dropout : float
        Dropout inside the blocks.
    bias : bool
        Whether the layers have biases.
    """

    def __init__(self, hidden: int, layers: int, kernel: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                CausalBlock(
                    SERIES if level == 0 else hidden, hidden, kernel, 2**level, dropout, bias
                )
                for level in range(layers)
            )
        )
        self.head = nn.Linear(hidden, 1, bias=bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Forecast from a batch of rows.

        Parameters
        ----------
        rows : torch.Tensor
            Lag-feature rows, shape (batch, SERIES x L).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1): from the last time step's channels.
        """
        return self.head(self.blocks(to_sequence(rows).transpose(1, 2))[:, :, -1])


class BeatsBlock(nn.Module):
    """
    One generic N-BEATS block: fully connected ReLU layers, then a backcast and a forecast.

    Parameters
    ----------
    inputs : int
        Width of the input window (SERIES x L).
    hidden : int
        Units per fully connected layer.
    layers : int
        Fully connected layers.
    dropout : float
        Dropout after each of them.
    bias : bool
        Whether the layers have biases.
    """

    def __init__(self, inputs: int, hidden: int, layers: int, dropout: float, bias: bool) -> None:
        super().__init__()
        stack: list[nn.Module] = []
        width = inputs
        for _ in range(layers):
            stack += [nn.Linear(width, hidden, bias=bias), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.stack = nn.Sequential(*stack)
        self.backcast = nn.Linear(hidden, inputs, bias=bias)
        self.forecast = nn.Linear(hidden, 1, bias=bias)

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the block.

        Parameters
        ----------
        window : torch.Tensor
            Shape (batch, inputs): what earlier blocks haven't explained yet.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The backcast, shape (batch, inputs), and this block's forecast, shape (batch, 1).
        """
        hidden = self.stack(window)
        return self.backcast(hidden), self.forecast(hidden)


class Beats(nn.Module):
    """
    A generic N-BEATS forecaster: blocks in a residual chain.

    Parameters
    ----------
    inputs : int
        Width of the input window (SERIES x L).
    hidden : int
        Units per fully connected layer.
    layers : int
        Fully connected layers per block.
    blocks : int
        Blocks in the chain.
    dropout : float
        Dropout after each fully connected layer.
    bias : bool
        Whether the layers have biases.

    Notes
    -----
    Each block sees what the earlier blocks' backcasts left of the window, and the forecast is
    the sum of every block's (Oreshkin et al. 2020). The window is both series' lags; the
    fully connected layers don't depend on their order.
    """

    def __init__(
        self, inputs: int, hidden: int, layers: int, blocks: int, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            BeatsBlock(inputs, hidden, layers, dropout, bias) for _ in range(blocks)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Forecast from a batch of rows.

        Parameters
        ----------
        rows : torch.Tensor
            Lag-feature rows, shape (batch, SERIES x L).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1).
        """
        residual, forecast = rows, torch.zeros(len(rows), 1)
        for block in self.blocks:
            backcast, step = block(residual)
            residual, forecast = residual - backcast, forecast + step
        return forecast


class DecompositionLinear(nn.Module):
    """
    A DLinear forecaster: moving-average trend and remainder, each through one linear layer.

    Parameters
    ----------
    inputs : int
        Width of the input window (SERIES x L).
    kernel : int
        Moving-average length, odd.
    bias : bool
        Whether the linear layers have biases.

    Notes
    -----
    As in Zeng et al. (2023): the trend is a centred moving average along time, with the
    window's first and last values repeated as padding so it keeps the window's length, and
    the seasonal-remainder is what the trend leaves. Both series' components feed one output.
    """

    def __init__(self, inputs: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(inputs, 1, bias=bias)
        self.remainder = nn.Linear(inputs, 1, bias=bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Forecast from a batch of rows.

        Parameters
        ----------
        rows : torch.Tensor
            Lag-feature rows, shape (batch, SERIES x L).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1).
        """
        sequence = to_sequence(rows)
        half = (self.kernel - 1) // 2
        padded = torch.cat(
            [sequence[:, :1].repeat(1, half, 1), sequence, sequence[:, -1:].repeat(1, half, 1)],
            dim=1,
        )
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)
        remainder = sequence - trend
        return self.trend(trend.flatten(1)) + self.remainder(remainder.flatten(1))


def build_sequence_module(architecture: str, inputs: int, settings: Any) -> nn.Module:
    """
    One sequence network, from a TorchNetwork's settings.

    Parameters
    ----------
    architecture : str
        One of SEQUENCE_ARCHITECTURES.
    inputs : int
        Width of the feature rows (SERIES x L).
    settings : TorchNetwork
        Supplies hidden_units, layers, kernel_size, blocks, dropout and use_bias.

    Returns
    -------
    torch.nn.Module
        The untrained network; its forward() maps rows (batch, inputs) to (batch, 1).

    Raises
    ------
    ValueError
        If `architecture` isn't one of SEQUENCE_ARCHITECTURES.
    """
    hidden, layers, dropout, bias = (
        settings.hidden_units,
        settings.layers,
        settings.dropout,
        settings.use_bias,
    )
    if architecture in ("lstm", "gru"):
        return Recurrent(architecture, hidden, layers, dropout, bias)
    if architecture == "tcn":
        return TemporalConvolution(hidden, layers, settings.kernel_size, dropout, bias)
    if architecture == "nbeats":
        return Beats(inputs, hidden, layers, settings.blocks, dropout, bias)
    if architecture == "dlinear":
        return DecompositionLinear(inputs, settings.kernel_size, bias)
    raise ValueError(f"unknown sequence network {architecture!r}")
