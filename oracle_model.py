"""Oracle model definitions — MLP, Transformer, and ConfidenceHead."""

import torch
import torch.nn as nn


class MLPOracle(nn.Module):
    """3-layer bottleneck MLP: LayerNorm → Linear(down) → GELU → Linear → GELU → Linear(up)."""

    def __init__(self, in_dim=768, small_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformerBlock(nn.Module):
    """Pre-LN Transformer block with self-attention and FFN, at reduced dimension."""

    def __init__(self, dim=96, heads=4, ff_mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        return residual + self.ffn(x)


class TransformerOracle(nn.Module):
    """Oracle with linear projection, N tiny transformer blocks, and linear projection back."""

    def __init__(self, in_dim=768, small_dim=96, num_blocks=1, heads=4):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, small_dim, bias=False)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(small_dim, heads) for _ in range(num_blocks)]
        )
        self.ln_out = nn.LayerNorm(small_dim)
        self.proj_up = nn.Linear(small_dim, in_dim, bias=False)

    def forward(self, x):
        x = self.proj_down(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.proj_up(x)


class ConfidenceHead(nn.Module):
    """Predicts per-token confidence of the oracle's prediction.

    Outputs a scalar in ``[0, 1]`` — higher means the oracle is more likely
    to have produced an accurate prediction. Used at inference time to decide
    whether to accept the oracle's output or fall back to the original layers.
    """

    def __init__(self, hidden_size: int, small_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-token confidence scores.

        Args:
            x: Oracle's input hidden state, shape ``(batch, seq, hidden)``.

        Returns:
            Confidence scores, shape ``(batch, seq, 1)``, values in ``[0, 1]``.
        """
        return self.net(x)


class MSEPredictionHead(nn.Module):
    """Predicts per-token normalized MSE of the oracle's prediction.

    This is a **regression** head: it directly predicts how large the oracle's
    error will be for each token. Lower predicted MSE → more reliable oracle.

    The predicted MSE is normalized by hidden_size so it's scale-invariant
    across different model sizes.
    """

    def __init__(self, hidden_size: int, small_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, small_dim),
            nn.GELU(),
            nn.Linear(small_dim, 1),
            nn.Softplus(),  # ensures positive output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-token predicted MSE, shape ``(batch, seq, 1)``."""
        return self.net(x)


def build_oracle(
    oracle_type: str,
    in_dim: int,
    small_dim: int,
    num_blocks: int = 1,
) -> nn.Module:
    """Factory: returns an ``MLPOracle`` or ``TransformerOracle``."""
    if oracle_type == "mlp":
        return MLPOracle(in_dim=in_dim, small_dim=small_dim)
    if oracle_type == "transformer":
        return TransformerOracle(
            in_dim=in_dim,
            small_dim=small_dim,
            num_blocks=num_blocks,
        )
    raise ValueError(f"Unsupported oracle_type: {oracle_type}")


def build_oracle_with_confidence(
    oracle_type: str,
    in_dim: int,
    small_dim: int,
    num_blocks: int = 1,
) -> tuple[nn.Module, ConfidenceHead]:
    """Factory that returns ``(oracle, confidence_head)``.

    The confidence head's hidden size is ``max(64, small_dim)``.
    """
    oracle = build_oracle(oracle_type, in_dim, small_dim, num_blocks)
    conf_head = ConfidenceHead(in_dim, max(64, small_dim))
    return oracle, conf_head


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
