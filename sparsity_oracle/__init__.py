"""Sparsity Oracle — exploit FFN activation sparsity for lossless compute savings."""

from .analyze_sparsity import analyze_ffn_sparsity, SparsityReport
from .sparse_ffn import SparseFFN, topk_activation

__all__ = ["analyze_ffn_sparsity", "SparsityReport", "SparseFFN", "topk_activation"]
