"""Benchmark FLOPs savings from sparsity exploitation.

Estimates the practical compute savings from FFN activation sparsity,
accounting for the overhead of the sparsity selector and sparse matmul.

Usage:
    python -m sparsity_oracle.benchmark --preset distilgpt2 --seq-len 4096
"""

import argparse
import json

import numpy as np

from sparsity_oracle.analyze_sparsity import analyze_ffn_sparsity
from config import build_experiment_config
from oracle_utils import setup_cpu_threads, setup_logging

import logging
logger = logging.getLogger(__name__)


def estimate_layer_flops(hidden_size: int, ffn_dim: int, seq_len: int, batch: int = 1) -> dict:
    """Estimate FLOPs for a single GPT-2 style transformer layer.

    GPT-2 layer: Attention + FFN

    Attention: Q,K,V projection + attention * softmax + output projection
    FFN: c_fc (up) + GELU + c_proj (down)

    Returns dict of FLOPs per component.
    """
    B, S, H, F = batch, seq_len, hidden_size, ffn_dim

    # Attention (simplified FLOPs estimate)
    # Q, K, V projections: each is (B,S,H) × (H,H) → 2*B*S*H^2
    # Actually for matmul: (M×K) × (K×N) ≈ 2*M*N*K FLOPs
    attn_proj = 3 * 2 * B * S * H * H  # Q, K, V projections

    # Attention scores: Q × K^T = (B,S,H) × (H,B,S) → but with multi-head this gets complex
    # Simplified: S^2 * H (inner products) × 2
    attn_scores = 2 * B * S * S * H

    # Softmax: negligible compared to matmul
    # Output projection: (B,S,H) × (H,H) → 2*B*S*H*H
    attn_out_proj = 2 * B * S * H * H

    attn_total = attn_proj + attn_scores + attn_out_proj

    # FFN
    # c_fc: (B,S,H) × (H,F) → 2*B*S*H*F
    ffn_up = 2 * B * S * H * F
    # GELU: ~5 * B*S*F (rough)
    ffn_act = 5 * B * S * F
    # c_proj: (B,S,F) × (F,H) → 2*B*S*F*H
    ffn_down = 2 * B * S * F * H
    ffn_total = ffn_up + ffn_act + ffn_down

    # Layer norm: 2 * 4 * B*S*H (rough)
    ln = 2 * 4 * B * S * H

    return {
        "attention": attn_total,
        "ffn": ffn_total,
        "ffn_up": ffn_up,
        "ffn_act": ffn_act,
        "ffn_down": ffn_down,
        "layer_norm": ln,
        "total": attn_total + ffn_total + ln,
    }


def estimate_sparse_ffn_flops(hidden_size: int, ffn_dim: int, seq_len: int,
                               k_frac: float = 0.15, batch: int = 1) -> dict:
    """Estimate FLOPs for sparse FFN: only compute top-k activations.

    The key insight: if we only keep k_frac of activations, we only need to
    compute k_frac of the down-projection columns. Additionally, we need a
    lightweight selector to identify which columns to keep.
    """
    B, S, H, F = batch, seq_len, hidden_size, ffn_dim
    K_dim = max(1, int(F * k_frac))

    # Up-projection: still full (we need all activations to find top-k)
    ffn_up = 2 * B * S * H * F

    # Selector: find top-k from F activations → O(F log k) per token
    # Rough estimate: 3 * B*S*F comparisons
    selector_flops = 3 * B * S * F

    # GELU: still full
    ffn_act = 5 * B * S * F

    # Sparse down-projection: only K_dim of F columns
    # (B,S,K_dim) × (K_dim,H) → 2*B*S*K_dim*H
    ffn_down_sparse = 2 * B * S * K_dim * H

    ffn_sparse_total = ffn_up + ffn_act + ffn_down_sparse + selector_flops

    return {
        "ffn_up": ffn_up,
        "selector": selector_flops,
        "ffn_act": ffn_act,
        "ffn_down": ffn_down_sparse,
        "ffn_sparse_total": ffn_sparse_total,
        "density": k_frac,
        "active_dim": K_dim,
    }


def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable form."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def main():
    parser = argparse.ArgumentParser(description="Benchmark FLOPs savings from FFN sparsity")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length for prefill")
    parser.add_argument("--samples", type=int, default=100, help="Samples for sparsity analysis")
    parser.add_argument("--k-frac", type=float, default=0.15, help="Fraction of activations to keep")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(preset=args.preset)

    # Run sparsity analysis
    logger.info("Running sparsity analysis (%d samples)...", args.samples)
    report = analyze_ffn_sparsity(
        model_name=cfg.model_name,
        max_samples=args.samples,
    )

    hidden_size = report.hidden_size
    seq_len = args.seq_len

    # Per-layer FLOPs
    total_dense = 0
    total_sparse = 0
    layer_details = []

    for l in report.layers:
        ffn_dim = l.ffn_dim
        dense = estimate_layer_flops(hidden_size, ffn_dim, seq_len)
        sparse = estimate_sparse_ffn_flops(hidden_size, ffn_dim, seq_len, args.k_frac)

        total_dense += dense["total"]
        total_sparse += (dense["attention"] + dense["layer_norm"] + sparse["ffn_sparse_total"])

        layer_details.append({
            "layer": l.layer_idx,
            "sparsity": round(l.mean_sparsity, 4),
            "dense_ffn_flops": dense["ffn"],
            "sparse_ffn_flops": sparse["ffn_sparse_total"],
            "ffn_saving_ratio": round(1 - sparse["ffn_sparse_total"] / dense["ffn"], 4),
            "total_saving_ratio": round(
                1 - (dense["attention"] + dense["layer_norm"] + sparse["ffn_sparse_total"]) / dense["total"], 4
            ),
        })

    saving_ratio = 1 - total_sparse / total_dense

    # Print
    print(f"\n{'='*70}")
    print(f"FFN Sparsity Benchmark: {cfg.model_name}")
    print(f"{'='*70}")
    print(f"Sequence length: {seq_len}  |  Keep top: {args.k_frac*100:.0f}%")
    print(f"Hidden size: {hidden_size}")
    print(f"Overall sparsity: {report.overall_mean_sparsity*100:.1f}%")
    print()
    print(f"Per-layer FLOPs (single token, batch=1):")
    print(f"  {'Layer':>5s} | {'Dense FLOPs':>12s} | {'Sparse FLOPs':>13s} | {'FFN Save':>8s} | {'Total Save':>9s}")
    print(f"  {'-'*60}")
    for d in layer_details:
        print(f"  {d['layer']:5d} | {format_flops(d['dense_ffn_flops']):>12s} | {format_flops(d['sparse_ffn_flops']):>13s} | {d['ffn_saving_ratio']*100:6.1f}% | {d['total_saving_ratio']*100:7.1f}%")

    print(f"\n  Dense total:  {format_flops(total_dense)}")
    print(f"  Sparse total: {format_flops(total_sparse)}")
    print(f"  Saving:       {saving_ratio*100:.1f}%")

    # MoE scenario estimate
    print(f"\n{'─'*50}")
    print("MoE model extrapolation (hypothetical):")
    print(f"  Assume 61 layers, hidden=7168, 40 MoE layers, seq_len={seq_len}")
    large_h = 7168
    large_f = 7168 * 4  # typical FFN dim = 4× hidden
    large_dense = estimate_layer_flops(large_h, large_f, seq_len)
    large_sparse = estimate_sparse_ffn_flops(large_h, large_f, seq_len, args.k_frac)
    moe_sparsity = report.overall_mean_sparsity  # assume similar sparsity
    n_total = 61
    n_moe = 40
    n_dense = 21

    total_moe_dense = n_total * large_dense["total"]
    total_moe_sparse = (
        n_dense * large_dense["total"] +
        n_moe * (large_dense["attention"] + large_dense["layer_norm"] + large_sparse["ffn_sparse_total"])
    )
    moe_saving = 1 - total_moe_sparse / total_moe_dense
    print(f"  Total dense:  {format_flops(total_moe_dense)}")
    print(f"  Total sparse: {format_flops(total_moe_sparse)}")
    print(f"  Saving:       {moe_saving*100:.1f}%")

    # Save
    output_path = args.output or f"benchmark_{cfg.model_short}_seq{seq_len}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": cfg.model_name,
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "keep_top_frac": args.k_frac,
            "overall_sparsity": round(report.overall_mean_sparsity, 4),
            "dense_total_flops": total_dense,
            "sparse_total_flops": total_sparse,
            "saving_ratio": round(saving_ratio, 4),
            "per_layer": layer_details,
            "moe_extrapolation_saving_ratio": round(moe_saving, 4),
        }, f, indent=2)
    logger.info("Benchmark saved to %s", output_path)


if __name__ == "__main__":
    main()
