"""
models.py
=========
Three sequence-classification / regression models built in PyTorch:

  1. RNN  – Vanilla Elman recurrent network
  2. LSTM – Long Short-Term Memory network
  3. GRU  – Gated Recurrent Unit network

All three share the same dual-head architecture:
  • Classification head → binary next-day direction (up / down)
  • Regression head     → predicted next-day return magnitude

Architecture details
--------------------
- Configurable hidden size, number of layers, and dropout
- LayerNorm on the final hidden state for training stability
- Separate linear heads for classification and regression
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# SHARED BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class _SequenceBase(nn.Module):
    """
    Abstract base: apply a recurrent core, then two output heads.
    Subclasses only need to define `self.rnn`.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Subclasses set self.rnn  ← the recurrent block
        self.rnn = None

        # Layer normalisation stabilises training for financial time series
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout between recurrent output and heads
        self.dropout = nn.Dropout(dropout)

        # Classification head: binary direction (up=1 / down=0)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),          # 2 logits for CrossEntropyLoss
        )

        # Regression head: predict return magnitude
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),          # scalar output
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the recurrent layer and extract the final hidden state.

        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        h_last : (batch, hidden_size)  – final time step representation
        """
        # rnn_out : (batch, seq_len, hidden_size)
        rnn_out, _ = self.rnn(x)
        h_last = rnn_out[:, -1, :]        # take the last time step
        h_last = self.layer_norm(h_last)
        h_last = self.dropout(h_last)
        return h_last

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        logits : (batch, 2)   – raw classification logits
        reg    : (batch,)     – regression prediction
        """
        h = self._encode(x)
        logits = self.cls_head(h)
        reg    = self.reg_head(h).squeeze(-1)
        return logits, reg


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: VANILLA RNN
# ─────────────────────────────────────────────────────────────────────────────

class RNNModel(_SequenceBase):
    """
    Vanilla Elman RNN with two output heads.

    Notes
    -----
    Vanilla RNNs suffer from vanishing/exploding gradients for long sequences.
    We mitigate this with gradient clipping in the trainer and a short SEQ_LEN.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__(n_features, hidden_size, num_layers, dropout)
        self.rnn = nn.RNN(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            nonlinearity  = "tanh",
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for stability."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def extra_repr(self) -> str:
        return f"RNN | hidden={self.hidden_size} layers={self.num_layers}"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(_SequenceBase):
    """
    LSTM with forget-gate bias initialised to 1 (standard best practice)
    to encourage the model to remember long-term dependencies early in training.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__(n_features, hidden_size, num_layers, dropout)
        self.rnn = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Override to handle LSTM's (h, c) output tuple."""
        out, (h_n, _) = self.rnn(x)
        h_last = out[:, -1, :]
        h_last = self.layer_norm(h_last)
        h_last = self.dropout(h_last)
        return h_last

    def extra_repr(self) -> str:
        return f"LSTM | hidden={self.hidden_size} layers={self.num_layers}"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: GRU
# ─────────────────────────────────────────────────────────────────────────────

class GRUModel(_SequenceBase):
    """
    GRU – similar capacity to LSTM with fewer parameters.
    Often trains faster and generalises well on short financial sequences.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__(n_features, hidden_size, num_layers, dropout)
        self.rnn = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def extra_repr(self) -> str:
        return f"GRU | hidden={self.hidden_size} layers={self.num_layers}"


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_all_models(
    n_features:  int,
    hidden_size: int = 64,
    num_layers:  int = 2,
    dropout:     float = 0.3,
) -> dict[str, nn.Module]:
    """
    Instantiate all three models with identical hyperparameters for fair comparison.

    Returns
    -------
    dict: {"RNN": model, "LSTM": model, "GRU": model}
    """
    return {
        "RNN":  RNNModel( n_features, hidden_size, num_layers, dropout),
        "LSTM": LSTMModel(n_features, hidden_size, num_layers, dropout),
        "GRU":  GRUModel( n_features, hidden_size, num_layers, dropout),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BATCH, SEQ, FEATS = 8, 10, 11   # 6 price + 5 sentiment
    x = torch.randn(BATCH, SEQ, FEATS)

    for name, model in build_all_models(FEATS).items():
        logits, reg = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:5s} | logits={tuple(logits.shape)} | reg={tuple(reg.shape)} | params={params:,}")
