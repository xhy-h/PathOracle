"""Top-k sparse FFN — only compute the most important activation columns."""

import torch
import torch.nn as nn


def topk_activation(
    x: torch.Tensor,
    act: torch.Tensor,
    k_frac: float = 0.15,
) -> torch.Tensor:
    """Keep only the top *k_frac* fraction of activations; zero out the rest.

    This is the core sparsity exploitation trick: if 85% of activations are near
    zero, we can zero them out without meaningfully changing the output, then
    skip computing their corresponding columns in ``down_proj``.

    Args:
        x: Original input to the FFN (not used in this simple version, but
           preserved for interface compatibility).
        act: GELU/ReLU activation output, shape ``(..., ffn_dim)``.
        k_frac: Fraction of activations to keep (e.g. 0.15 = keep top 15%).

    Returns:
        Activation tensor with small values zeroed out. The number of non-zero
        entries is ``ceil(k_frac * ffn_dim)`` per position.
    """
    k = max(1, int(act.size(-1) * k_frac))
    threshold = act.abs().kthvalue(act.size(-1) - k + 1, dim=-1).values.unsqueeze(-1)
    return torch.where(act.abs() >= threshold, act, torch.tensor(0.0, device=act.device))


class SparseFFN(nn.Module):
    """A wrapper that makes any FFN module sparse by zeroing small activations.

    This does NOT reduce FLOPs by itself (the matrix multiply is still dense).
    It is used for *evaluation*: to measure how much quality is preserved when
    activations are sparsified. A production implementation would replace the
    ``down_proj`` with a sparse matrix multiply that skips zero columns.

    Args:
        ffn: The original FFN module (e.g. ``model.transformer.h[i].mlp``).
        k_frac: Fraction of activations to retain.
    """

    def __init__(self, ffn: nn.Module, k_frac: float = 0.15):
        super().__init__()
        self.ffn = ffn
        self.k_frac = k_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with activation sparsification, but still dense computation.

        For GPT-2 style FFN::
            Linear(c_fc) → GELU(act) → topk_sparsify → Linear(c_proj)

        Returns:
            Output tensor of the same shape as input.
        """
        # GPT-2 FFN structure
        if hasattr(self.ffn, 'c_fc'):
            mid = self.ffn.c_fc(x)
            act = self.ffn.act(mid)
            sparse_act = topk_activation(x, act, self.k_frac)
            return self.ffn.c_proj(sparse_act)
        # Generic FFN: assume it's a Sequential of Linear → Act → Linear
        elif isinstance(self.ffn, nn.Sequential) and len(self.ffn) == 3:
            mid = self.ffn[0](x)
            act = self.ffn[1](mid)
            sparse_act = topk_activation(x, act, self.k_frac)
            return self.ffn[2](sparse_act)
        else:
            raise NotImplementedError(f"Unsupported FFN structure: {type(self.ffn)}")
