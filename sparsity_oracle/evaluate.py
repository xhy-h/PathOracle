"""Evaluate quality impact of sparse FFN — measure cosine similarity vs full FFN.

Usage:
    python -m sparsity_oracle.evaluate --preset distilgpt2 --topk 0.15 --samples 100
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from config import build_experiment_config, validate_model_shape
from oracle_utils import setup_cpu_threads, setup_logging, setup_tokenizer
from sparsity_oracle.sparse_ffn import SparseFFN

import logging
logger = logging.getLogger(__name__)


@torch.no_grad()
def compare_ffn_outputs(
    model,
    input_ids,
    k_frac: float = 0.15,
) -> list[dict]:
    """Compare full FFN vs sparse FFN outputs for every layer.

    Returns:
        List of dicts with keys: layer, cosine_sim, mse, sparsity.
    """
    hidden = model.transformer.wte(input_ids) + \
             model.transformer.wpe(torch.arange(input_ids.size(1)).unsqueeze(0))
    hidden = model.transformer.drop(hidden)

    results = []
    for i, layer in enumerate(model.transformer.h):
        # Run Attention (unchanged)
        residual = hidden
        h_attn = layer.ln_1(hidden)
        h_attn = layer.attn(h_attn)[0]
        hidden = residual + h_attn

        # Run both full and sparse FFN
        residual = hidden
        h_ffn = layer.ln_2(hidden)

        # Full FFN
        mlp = layer.mlp
        full_mid = mlp.c_fc(h_ffn)
        full_act = mlp.act(full_mid)
        full_out = residual + mlp.c_proj(full_act)

        # Sparse FFN (top-k)
        sparse = SparseFFN(mlp, k_frac=k_frac)
        sparse_out = residual + sparse(h_ffn)

        # Compare
        diff = full_out - sparse_out
        cos_sim = torch.nn.functional.cosine_similarity(
            full_out.flatten(1), sparse_out.flatten(1), dim=-1
        ).mean().item()
        mse = (diff ** 2).mean().item()

        # Actual sparsity achieved
        sparse_act = mlp.act(mlp.c_fc(h_ffn))
        abs_act = sparse_act.abs()
        threshold = abs_act.kthvalue(
            max(1, int(abs_act.size(-1) * (1 - k_frac))),
            dim=-1,
        ).values.unsqueeze(-1)
        achieved_sparsity = (abs_act < threshold).float().mean().item()

        results.append({
            "layer": i,
            "cosine_sim": round(cos_sim, 6),
            "mse": round(mse, 8),
            "target_sparsity": k_frac,
            "achieved_sparsity": round(achieved_sparsity, 4),
        })

        hidden = full_out  # always use full output for next layer

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse FFN quality")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--topk", type=float, default=0.15,
                        help="Fraction of activations to keep (e.g. 0.15 = top 15%%)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(
        preset=args.preset,
        max_length=args.max_length,
    )

    logger.info("Loading model %s ...", cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.eval()
    tokenizer = setup_tokenizer(cfg.model_name)
    validate_model_shape(model, cfg)

    logger.info("Loading dataset ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

    # Accumulate metrics
    all_results = []
    samples = 0
    pbar = tqdm(total=args.samples, desc="Evaluating")
    for item in dataset:
        if samples >= args.samples:
            break
        text = item.get("text", "")
        if len(text.strip()) <= 20:
            continue
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=args.max_length, truncation=True)
        if inputs["input_ids"].size(1) < 10:
            continue

        layer_results = compare_ffn_outputs(model, inputs["input_ids"], args.topk)
        all_results.append(layer_results)
        samples += 1
        pbar.update(1)
    pbar.close()

    # Aggregate per-layer
    num_layers = len(all_results[0])
    agg = []
    for layer_idx in range(num_layers):
        cosines = [r[layer_idx]["cosine_sim"] for r in all_results]
        mses = [r[layer_idx]["mse"] for r in all_results]
        sparsities = [r[layer_idx]["achieved_sparsity"] for r in all_results]
        agg.append({
            "layer": layer_idx,
            "cosine_sim_mean": round(np.mean(cosines), 6),
            "cosine_sim_min": round(np.min(cosines), 6),
            "mse_mean": round(np.mean(mses), 8),
            "achieved_sparsity_mean": round(np.mean(sparsities), 4),
        })

    avg_cosine = np.mean([a["cosine_sim_mean"] for a in agg])

    # Print
    print(f"\n{'='*70}")
    print(f"Sparse FFN Evaluation: {cfg.model_name}  keep_top={args.topk}")
    print(f"{'='*70}")
    print(f"Samples: {samples}  Max length: {args.max_length}")
    print(f"Average cosine similarity across all layers: {avg_cosine:.6f}")
    print(f"\nPer-layer:")
    print(f"  Layer | Cosine Sim (mean) | Cosine Sim (min) | MSE (mean) | Sparsity")
    print(f"  {'-'*55}")
    for a in agg:
        print(f"  {a['layer']:5d} | {a['cosine_sim_mean']:14.6f} | {a['cosine_sim_min']:13.6f} | "
              f"{a['mse_mean']:9.2e} | {a['achieved_sparsity_mean']:.1%}")
    print(f"  {'-'*55}")

    # Save
    output_path = args.output or f"sparse_eval_{cfg.model_short}_top{int(args.topk*100)}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": cfg.model_name,
            "keep_top_frac": args.topk,
            "samples": samples,
            "avg_cosine_similarity": round(avg_cosine, 6),
            "per_layer": agg,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
