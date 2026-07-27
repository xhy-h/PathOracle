"""Step ②: extract (early_hidden, target_hidden) pairs from a pretrained model."""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from config import build_experiment_config, validate_model_shape
from oracle_utils import embed_inputs, setup_cpu_threads, setup_logging, setup_tokenizer

logger = logging.getLogger(__name__)


def collect_pair(model, input_ids, cfg):
    """Run early layers, record hidden state, run skipped layers, record target."""
    hidden = embed_inputs(model, input_ids)

    for layer_idx in range(cfg.early_count):
        hidden = model.transformer.h[layer_idx](hidden)[0]
    early = hidden.detach().squeeze(0).cpu().numpy()

    for layer_idx in range(cfg.early_count, cfg.target_layer_start):
        hidden = model.transformer.h[layer_idx](hidden)[0]
    target = hidden.detach().squeeze(0).cpu().numpy()

    return early.astype(np.float16), target.astype(np.float16)


def count_existing_pairs(data_dir: Path) -> int:
    """Count how many ``pair_*.npz`` files already exist in *data_dir*."""
    if not data_dir.exists():
        return 0
    candidates = [f for f in os.listdir(data_dir) if f.startswith("pair_") and f.endswith(".npz")]
    return len(candidates)


def main():
    parser = argparse.ArgumentParser(description="Collect hidden-state pairs for oracle training")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(
        preset=args.preset,
        samples=args.samples,
        max_length=args.max_length,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        streaming=args.streaming,
        run_tag=args.run_tag,
        data_dir=args.data_dir,
    )
    samples = cfg.collect_samples
    max_length = cfg.max_length
    data_dir = Path(cfg.data_dir)

    # Clear existing data if requested
    if args.clear and data_dir.exists():
        logger.warning("Clearing existing data directory: %s", data_dir)
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Incremental: start from existing count
    existing = count_existing_pairs(data_dir)
    if existing > 0:
        logger.info("Found %d existing pairs, resuming collection", existing)

    # Load model
    logger.info("Loading model %s ...", cfg.model_name)
    tokenizer = setup_tokenizer(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.eval()
    validate_model_shape(model, cfg)

    # Load dataset
    try:
        if cfg.dataset_config:
            dataset = load_dataset(
                cfg.dataset_name,
                cfg.dataset_config,
                split=cfg.dataset_split,
                streaming=cfg.streaming,
            )
        else:
            dataset = load_dataset(
                cfg.dataset_name,
                split=cfg.dataset_split,
                streaming=cfg.streaming,
            )
    except Exception as exc:
        logger.error("Failed to load dataset '%s' with config '%s': %s",
                      cfg.dataset_name, cfg.dataset_config, exc)
        sys.exit(1)

    collected = existing
    pbar = tqdm(total=samples, desc="collect", initial=collected)
    with torch.no_grad():
        for item in dataset:
            if collected >= samples:
                break

            text = item.get("text", "")
            if len(text.strip()) <= 50:
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = inputs["input_ids"]
            if input_ids.size(1) < 10:
                continue

            early, target = collect_pair(model, input_ids, cfg)
            out_path = data_dir / f"pair_{collected:06d}.npz"
            np.savez_compressed(out_path, early=early, target=target)
            collected += 1
            pbar.update(1)

    pbar.close()

    metadata = {
        "preset": args.preset,
        "model_name": cfg.model_name,
        "hidden_size": cfg.hidden_size,
        "total_layers": cfg.total_layers,
        "early_count": cfg.early_count,
        "target_layer_start": cfg.target_layer_start,
        "late_count": cfg.late_count,
        "max_length": max_length,
        "samples": collected,
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "dataset_split": cfg.dataset_split,
        "streaming": cfg.streaming,
        "run_tag": cfg.run_tag,
        "resumed": existing > 0,
    }
    # Atomic write via temp file
    meta_tmp = data_dir / "metadata.json.tmp"
    meta_dst = data_dir / "metadata.json"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    meta_tmp.replace(meta_dst)

    logger.info("collected=%d  data_dir=%s", collected, data_dir)


if __name__ == "__main__":
    main()
