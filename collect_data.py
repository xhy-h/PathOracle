import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import build_experiment_config, validate_model_shape


def embed_inputs(model, input_ids):
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(position_ids)
    return model.transformer.drop(hidden)


def collect_pair(model, input_ids, cfg):
    hidden = embed_inputs(model, input_ids)

    for layer_idx in range(cfg.early_count):
        hidden = model.transformer.h[layer_idx](hidden)[0]
    early = hidden.detach().squeeze(0).cpu().numpy()

    for layer_idx in range(cfg.early_count, cfg.target_layer_start):
        hidden = model.transformer.h[layer_idx](hidden)[0]
    target = hidden.detach().squeeze(0).cpu().numpy()

    return early.astype(np.float16), target.astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

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

    if args.clear and data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.eval()
    validate_model_shape(model, cfg)

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

    collected = 0
    pbar = tqdm(total=samples, desc="collect")
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
    }
    with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"collected={collected}")
    print(f"data_dir={data_dir}")


if __name__ == "__main__":
    main()
