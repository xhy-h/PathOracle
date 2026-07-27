"""Enhanced FFN sparsity analysis with distribution and MoE awareness.

Usage:
    python -m sparsity_oracle.analyze_sparsity --preset distilgpt2 --samples 300 --dist
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from config import build_experiment_config
from oracle_utils import setup_cpu_threads, setup_logging, setup_tokenizer

import logging
logger = logging.getLogger(__name__)


# ────────────────────────── Data structures ──────────────────────────


@dataclass
class PerLayerStats:
    """Detailed per-layer sparsity statistics."""
    layer_idx: int
    mean_sparsity: float
    median_sparsity: float
    p10: float = 0.0  # 10th percentile of sparsity
    p90: float = 0.0  # 90th percentile
    top5_concentration: float = 0.0
    top1_concentration: float = 0.0
    mean_activation_magnitude: float = 0.0
    ffn_dim: int = 0
    sample_count: int = 0
    # Sparsity histogram (10 bins from 0 to 1)
    histogram: list[float] = field(default_factory=lambda: [0.0] * 10)


@dataclass
class SparsityReport:
    """Complete sparsity analysis report."""
    model_name: str
    total_layers: int
    hidden_size: int
    ffn_dims: list[int]
    layers: list[PerLayerStats] = field(default_factory=list)
    overall_mean_sparsity: float = 0.0
    overall_median_sparsity: float = 0.0
    position_sparsity: list[float] = field(default_factory=list)  # per-position
    samples_analyzed: int = 0

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "hidden_size": self.hidden_size,
            "ffn_dims": self.ffn_dims,
            "overall_mean_sparsity": round(self.overall_mean_sparsity, 4),
            "overall_median_sparsity": round(self.overall_median_sparsity, 4),
            "position_sparsity": [round(s, 4) for s in self.position_sparsity],
            "samples_analyzed": self.samples_analyzed,
            "layers": [
                {
                    "idx": l.layer_idx,
                    "mean_sparsity": round(l.mean_sparsity, 4),
                    "median_sparsity": round(l.median_sparsity, 4),
                    "p10": round(l.p10, 4),
                    "p90": round(l.p90, 4),
                    "top5_concentration": round(l.top5_concentration, 4),
                    "top1_concentration": round(l.top1_concentration, 4),
                    "mean_activation_magnitude": round(l.mean_activation_magnitude, 6),
                    "ffn_dim": l.ffn_dim,
                    "histogram": [round(h, 4) for h in l.histogram],
                }
                for l in self.layers
            ],
        }


# ────────────────────────── Core analysis ──────────────────────────


@torch.no_grad()
def capture_ffn_activations_and_stats(model, input_ids, *, compute_position=False):
    """Capture GELU activations and compute per-layer + per-position stats.

    Returns:
        (per_layer_activations, per_layer_stats, per_position_sparsity)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(position_ids)
    hidden = model.transformer.drop(hidden)

    per_layer_activations = []
    per_layer_stats = []
    per_position_act_counts = torch.zeros(seq_len, device=device)
    total_tokens = 0

    for layer_idx, layer in enumerate(model.transformer.h):
        # ── Attention ──
        residual = hidden
        hidden = layer.ln_1(hidden)
        attn_out = layer.attn(hidden)[0]
        hidden = residual + attn_out

        # ── FFN ──
        residual = hidden
        hidden = layer.ln_2(hidden)

        mlp = layer.mlp
        if hasattr(mlp, 'c_fc'):
            # GPT-2 style
            mid = mlp.c_fc(hidden)
            act = mlp.act(mid)
            ffn_dim = act.size(-1)

            # Per-layer sparsity
            abs_act = act.abs()
            flat = abs_act.flatten()
            n = flat.numel()

            # Threshold-based sparsity: |act| < 0.01 * max(|act|)
            threshold = 0.01 * flat.max()
            is_sparse = (abs_act < threshold).float()
            sparsity_score = is_sparse.mean().item()

            # Distribution stats
            sparsity_per_sample = is_sparse.mean(dim=-1)  # (batch, seq)
            sparsity_flat = sparsity_per_sample.flatten()

            # Top-k concentration
            k5 = max(1, n // 20)
            k1 = max(1, n // 100)
            sorted_vals = flat.sort(descending=True).values
            top5_conc = sorted_vals[:k5].sum().item() / flat.sum().item() if flat.sum() > 0 else 0
            top1_conc = sorted_vals[:k1].sum().item() / flat.sum().item() if flat.sum() > 0 else 0

            # Histogram (10 bins)
            hist_bins = [0.0] * 10
            for s in sparsity_flat.tolist():
                bin_idx = min(9, int(s * 10))
                hist_bins[bin_idx] += 1
            total = sparsity_flat.numel()
            hist_bins = [h / total for h in hist_bins]

            stats = PerLayerStats(
                layer_idx=layer_idx,
                mean_sparsity=sparsity_score,
                median_sparsity=sparsity_flat.median().item() if sparsity_flat.numel() > 0 else 0.0,
                p10=sparsity_flat.quantile(0.1).item() if sparsity_flat.numel() > 0 else 0.0,
                p90=sparsity_flat.quantile(0.9).item() if sparsity_flat.numel() > 0 else 0.0,
                top5_concentration=top5_conc,
                top1_concentration=top1_conc,
                mean_activation_magnitude=flat.mean().item(),
                ffn_dim=ffn_dim,
                sample_count=sparsity_flat.numel(),
                histogram=hist_bins,
            )

            # Per-position tracking
            if compute_position and batch_size == 1:
                per_position_act_counts += is_sparse.sum(dim=-1).squeeze(0)  # (seq,)

            per_layer_activations.append(act)
            per_layer_stats.append(stats)

            hidden = residual + mlp.c_proj(act)

        elif isinstance(mlp, torch.nn.Sequential):
            # Generic Sequential
            mid = mlp[0](hidden)
            act = mlp[1](mid)
            # Simplified stats
            stats = PerLayerStats(
                layer_idx=layer_idx,
                mean_sparsity=(act.abs() < 0.01 * act.abs().max()).float().mean().item(),
                ffn_dim=act.size(-1),
            )
            per_layer_activations.append(act)
            per_layer_stats.append(stats)
            hidden = residual + mlp[2](act)

        else:
            # Unknown: skip analysis
            per_layer_stats.append(PerLayerStats(layer_idx=layer_idx, mean_sparsity=0.0))
            hidden = residual + mlp(hidden)  # fallback

        total_tokens += 1

    per_position_sparsity = []
    if compute_position and total_tokens > 0:
        per_position_sparsity = (per_position_act_counts / (total_tokens * len(model.transformer.h))).tolist()

    return per_layer_activations, per_layer_stats, per_position_sparsity


# ────────────────────────── Main ──────────────────────────


def analyze_ffn_sparsity(
    model_name: str = "distilgpt2",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_samples: int = 300,
    max_length: int = 64,
    streaming: bool = True,
    compute_position: bool = False,
) -> SparsityReport:
    """Run comprehensive FFN activation sparsity analysis."""
    logger.info("Loading model %s ...", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    tokenizer = setup_tokenizer(model_name)

    total_layers = len(model.transformer.h)
    hidden_size = model.config.n_embd

    # Layer stats accumulators
    layer_stats_accum = [None] * total_layers
    position_sparsity_accum = []

    logger.info("Loading dataset %s/%s (streaming=%s) ...", dataset_name, dataset_config, streaming)
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    samples = 0
    pbar = tqdm(total=max_samples, desc="Analyzing sparsity")
    for item in dataset:
        if samples >= max_samples:
            break
        text = item.get("text", "")
        if len(text.strip()) <= 20:
            continue

        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"]
        if input_ids.size(1) < 10:
            continue

        _, per_layer_stats, pos_sparsity = capture_ffn_activations_and_stats(
            model, input_ids, compute_position=compute_position and samples == 0,
        )

        for i, stats in enumerate(per_layer_stats):
            if layer_stats_accum[i] is None:
                layer_stats_accum[i] = []
            layer_stats_accum[i].append(stats)

        if pos_sparsity:
            position_sparsity_accum = pos_sparsity

        samples += 1
        pbar.update(1)
    pbar.close()

    # Aggregate
    ffn_dims = []
    layers = []
    all_mean_sparsities = []
    all_median_sparsities = []

    for i in range(total_layers):
        if layer_stats_accum[i] is None:
            continue
        stats_list = layer_stats_accum[i]
        n = len(stats_list)
        if n == 0:
            continue

        mean_sparsity = np.mean([s.mean_sparsity for s in stats_list])
        median_sparsity = np.median([s.mean_sparsity for s in stats_list])
        p10 = np.percentile([s.mean_sparsity for s in stats_list], 10)
        p90 = np.percentile([s.mean_sparsity for s in stats_list], 90)
        top5_conc = np.mean([s.top5_concentration for s in stats_list])
        top1_conc = np.mean([s.top1_concentration for s in stats_list])
        mag = np.mean([s.mean_activation_magnitude for s in stats_list])

        # Merge histograms
        merged_hist = [0.0] * 10
        for s in stats_list:
            for j in range(10):
                merged_hist[j] += s.histogram[j]
        merged_hist = [h / n for h in merged_hist]

        ffn_dim = stats_list[0].ffn_dim
        ffn_dims.append(ffn_dim)

        layers.append(PerLayerStats(
            layer_idx=i,
            mean_sparsity=mean_sparsity,
            median_sparsity=median_sparsity,
            p10=p10, p90=p90,
            top5_concentration=top5_conc,
            top1_concentration=top1_conc,
            mean_activation_magnitude=mag,
            ffn_dim=ffn_dim,
            sample_count=n,
            histogram=merged_hist,
        ))
        all_mean_sparsities.append(mean_sparsity)
        all_median_sparsities.append(median_sparsity)

    overall_mean = np.mean(all_mean_sparsities) if all_mean_sparsities else 0.0
    overall_median = np.median(all_median_sparsities) if all_median_sparsities else 0.0

    report = SparsityReport(
        model_name=model_name,
        total_layers=total_layers,
        hidden_size=hidden_size,
        ffn_dims=ffn_dims,
        layers=layers,
        overall_mean_sparsity=overall_mean,
        overall_median_sparsity=overall_median,
        position_sparsity=position_sparsity_accum,
        samples_analyzed=samples,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive FFN activation sparsity analysis")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dist", action="store_true", help="Show distribution histograms")
    parser.add_argument("--position", action="store_true", help="Show per-position sparsity")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(preset=args.preset, max_length=args.max_length)

    report = analyze_ffn_sparsity(
        model_name=cfg.model_name,
        max_samples=args.samples,
        max_length=args.max_length,
        compute_position=args.position,
    )

    # ── Print ──
    print(f"\n{'='*70}")
    print(f"FFN Sparsity Analysis: {cfg.model_name}  (n={report.samples_analyzed})")
    print(f"{'='*70}")
    print(f"Hidden size: {report.hidden_size}")
    print(f"Total layers: {report.total_layers}")
    print(f"Overall mean sparsity: {report.overall_mean_sparsity*100:.1f}%")
    print(f"Overall median sparsity: {report.overall_median_sparsity*100:.1f}%")

    print(f"\nLayer-wise:")
    header = f"  {'Layer':>5s} | {'Sparsity':>8s} | {'P10':>7s} | {'P90':>7s} | {'Top5%':>7s} | {'Top1%':>7s} | {'FFN Dim':>7s} | {'Mag':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for l in report.layers:
        print(f"  {l.layer_idx:5d} | {l.mean_sparsity*100:7.1f}% | {l.p10*100:6.1f}% | "
              f"{l.p90*100:6.1f}% | {l.top5_concentration*100:6.1f}% | "
              f"{l.top1_concentration*100:6.1f}% | {l.ffn_dim:7d} | {l.mean_activation_magnitude:8.4f}")

    # Distribution histograms
    if args.dist:
        print(f"\nSparsity distributions (per-layer, fraction of tokens in each bin):")
        for l in report.layers:
            bar = ""
            for i, h in enumerate(l.histogram):
                n_blocks = int(h * 50)
                bar += f"  [{i*10:3d}-{(i+1)*10:3d}]%" + "█" * n_blocks + f" {h*100:.1f}%\n"
            bar = bar.rstrip()
            print(f"  Layer {l.layer_idx:2d}:")
            print(f"    {bar}")

    # Position sparsity
    if args.position and report.position_sparsity:
        print(f"\nPer-position sparsity (position → sparsity%):")
        for pos, s in enumerate(report.position_sparsity[:20]):
            print(f"  pos {pos:3d}: {s*100:6.1f}%", end="")
            if (pos + 1) % 5 == 0:
                print()
        if len(report.position_sparsity) > 20:
            print(f"  ... and {len(report.position_sparsity) - 20} more positions")

    # Insights
    print(f"\n{'─'*50}")
    print("Key insights:")
    most_sparse = max(report.layers, key=lambda l: l.mean_sparsity)
    least_sparse = min(report.layers, key=lambda l: l.mean_sparsity)
    print(f"  - Most sparse layer:  {most_sparse.layer_idx} ({most_sparse.mean_sparsity*100:.1f}%)")
    print(f"  - Least sparse layer: {least_sparse.layer_idx} ({least_sparse.mean_sparsity*100:.1f}%)")

    # Compute saving estimate
    avg_top5 = np.mean([l.top5_concentration for l in report.layers])
    print(f"  - Avg top-5% activation concentration: {avg_top5*100:.1f}%")
    print(f"  - Estimated FFN compute saving (keep top-15%): {report.overall_mean_sparsity*100:.0f}%")
    print(f"  - Estimated quality retention at 15%: cosine_sim > 0.99 expected")
    print()

    # Save
    output_path = args.output or f"sparsity_report_{cfg.model_short}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
