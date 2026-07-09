import torch
import torch.nn as nn


class MLPOracle(nn.Module):
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


def build_oracle(
    oracle_type: str,
    in_dim: int,
    small_dim: int,
    num_blocks: int = 1,
) -> nn.Module:
    if oracle_type == "mlp":
        return MLPOracle(in_dim=in_dim, small_dim=small_dim)
    if oracle_type == "transformer":
        return TransformerOracle(
            in_dim=in_dim,
            small_dim=small_dim,
            num_blocks=num_blocks,
        )
    raise ValueError(f"Unsupported oracle_type: {oracle_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
