"""Step ③: train an oracle (MLP or Transformer) to predict skipped hidden states."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import build_experiment_config
from oracle_model import build_oracle, build_oracle_with_confidence, count_parameters, MSEPredictionHead
from oracle_utils import setup_cpu_threads, setup_logging

logger = logging.getLogger(__name__)


def resolve_resume_checkpoint(
    checkpoint_path: str,
    resume: bool = False,
    resume_from: str | None = None,
) -> Path | None:
    """Determine the checkpoint to resume from, or ``None`` for a fresh run.

    Resolution order:
    1. If ``resume_from`` is given: treat it as a path or a run-tag.
    2. If ``resume`` is True: use ``checkpoint_path``.
    3. Otherwise return ``None`` (fresh training).

    Args:
        checkpoint_path: Default checkpoint path (from config).
        resume: Use the default checkpoint path.
        resume_from: Explicit checkpoint path or run-tag.

    Returns:
        Resolved ``Path`` or ``None``.
    """
    if resume_from:
        source = Path(resume_from)
        if source.suffix == ".pth" or source.exists():
            return source
        return Path(f"oracle_{resume_from}.pth")
    if resume:
        return Path(checkpoint_path)
    return None


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    reset_optimizer: bool = False,
) -> tuple[list[dict], int]:
    """Load model weights and optionally optimizer/scheduler state from a checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint.
        model: Model to load state into.
        optimizer: Optimizer (state loaded unless ``reset_optimizer`` is True).
        scheduler: LR scheduler (state loaded unless ``reset_optimizer`` is True).
        reset_optimizer: If True, skip optimizer/scheduler state (for fine-tuning).

    Returns:
        Tuple of ``(history, completed_epochs)``.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])

    if not reset_optimizer and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if not reset_optimizer and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    history = list(payload.get("history", []))
    return history, len(history)


class HiddenStateDataset(Dataset):
    """Loads ``pair_*.npz`` files with optional length truncation."""

    def __init__(self, data_dir: str, max_len: int):
        self.data_dir = Path(data_dir)
        self.files = sorted(
            f for f in os.listdir(self.data_dir)
            if f.startswith("pair_") and f.endswith(".npz")
        )
        if not self.files:
            raise FileNotFoundError(f"No pair_*.npz files found in {self.data_dir}")
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.data_dir / self.files[idx])
        # Validate that the expected keys exist
        if "early" not in data or "target" not in data:
            raise KeyError(
                f"Expected keys 'early' and 'target' in {self.files[idx]}, "
                f"got {list(data.keys())}"
            )
        early = torch.from_numpy(data["early"].astype(np.float32))[: self.max_len]
        target = torch.from_numpy(data["target"].astype(np.float32))[: self.max_len]
        return early, target


def collate_fn(batch):
    """Zero-pad variable-length sequences with a boolean mask."""
    early_list, target_list = zip(*batch)
    batch_size = len(batch)
    max_len = max(item.size(0) for item in early_list)
    hidden_size = early_list[0].size(1)

    early_pad = torch.zeros(batch_size, max_len, hidden_size)
    target_pad = torch.zeros(batch_size, max_len, hidden_size)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for idx, (early, target) in enumerate(zip(early_list, target_list)):
        length = early.size(0)
        early_pad[idx, :length] = early
        target_pad[idx, :length] = target
        mask[idx, :length] = True

    return early_pad, target_pad, mask


def main():
    parser = argparse.ArgumentParser(description="Train an oracle network")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--oracle-type", default=None, choices=["mlp", "transformer"])
    parser.add_argument("--small-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--cos-weight", type=float, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--confidence-coeff", type=float, default=None)
    parser.add_argument("--no-confidence", action="store_true", help="Disable confidence head training")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(
        preset=args.preset,
        oracle_type=args.oracle_type,
        small_dim=args.small_dim,
        num_blocks=args.num_blocks,
        cos_weight=args.cos_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        run_tag=args.run_tag,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
    )
    oracle_type = cfg.oracle_type
    data_dir = cfg.data_dir
    checkpoint = cfg.checkpoint_path
    epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    max_len = args.max_len or cfg.max_length
    lr = cfg.learning_rate

    dataset = HiddenStateDataset(data_dir, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    confidence_enabled = not args.no_confidence
    confidence_head_type = "mse_regression"  # Options: "binary", "mse_regression"

    if confidence_enabled and confidence_head_type == "mse_regression":
        model = build_oracle(oracle_type, cfg.hidden_size, cfg.small_dim, cfg.num_blocks)
        confidence_head = MSEPredictionHead(cfg.hidden_size, max(64, cfg.small_dim))
        all_params = list(model.parameters()) + list(confidence_head.parameters())
    elif confidence_enabled:
        model, confidence_head = build_oracle_with_confidence(
            oracle_type, cfg.hidden_size, cfg.small_dim, cfg.num_blocks,
        )
        all_params = list(model.parameters()) + list(confidence_head.parameters())
    else:
        model = build_oracle(oracle_type, cfg.hidden_size, cfg.small_dim, cfg.num_blocks)
        all_params = model.parameters()
    
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_path=checkpoint,
        resume=args.resume,
        resume_from=args.resume_from,
    )
    history = []
    start_epoch = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        history, start_epoch = load_resume_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            reset_optimizer=args.reset_optimizer,
        )
        if start_epoch >= epochs:
            raise ValueError(
                f"Checkpoint already has {start_epoch} epochs; requested total epochs={epochs}"
            )
        logger.info("Resumed from %s at epoch %d", resume_checkpoint, start_epoch)

    logger.info(
        "oracle_type=%s  small_dim=%d  num_blocks=%d  cos_weight=%.2f  confidence=%s",
        oracle_type, cfg.small_dim, cfg.num_blocks, cfg.cos_weight, confidence_enabled,
    )
    logger.info("oracle_parameters=%d  samples=%d", count_parameters(model), len(dataset))

    for epoch in range(start_epoch, epochs):
        model.train()
        total_mse = 0.0
        total_cosine = 0.0
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")

        for early, target, mask in pbar:
            pred = model(early)
            pred_masked = pred[mask]
            target_masked = target[mask]

            mse = loss_fn(pred_masked, target_masked)
            cosine_sim = nn.functional.cosine_similarity(
                pred_masked, target_masked, dim=-1
            ).mean()
            loss = mse + cfg.cos_weight * (1.0 - cosine_sim)

            # Confidence head training
            conf_loss = torch.tensor(0.0)
            if confidence_enabled and confidence_head_type == "mse_regression":
                # Regression: predict actual per-token MSE, normalized by hidden_size
                with torch.no_grad():
                    per_token_mse = (pred_masked - target_masked).pow(2).mean(dim=-1, keepdim=True)  # (non_pad, 1)
                pred_mse = confidence_head(early)[mask]  # (non_pad, 1)
                conf_loss = nn.functional.mse_loss(pred_mse, per_token_mse)
                loss = loss + cfg.confidence_coeff * conf_loss
            elif confidence_enabled:
                # Binary: predict if MSE < median
                with torch.no_grad():
                    per_token_mse = (pred_masked - target_masked).pow(2).mean(dim=-1, keepdim=True)
                    median_mse = per_token_mse.median()
                    conf_labels = (per_token_mse < median_mse).float()
                conf_scores = confidence_head(early)[mask]
                conf_loss = nn.functional.binary_cross_entropy(conf_scores, conf_labels)
                loss = loss + cfg.confidence_coeff * conf_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_mse += mse.item()
            total_cosine += cosine_sim.item()
            postfix = {"mse": f"{mse.item():.4f}", "cos": f"{cosine_sim.item():.4f}"}
            if confidence_enabled:
                postfix["conf"] = f"{conf_loss.item():.4f}"
            pbar.set_postfix(postfix)

        scheduler.step()
        avg_mse = total_mse / len(dataloader)
        avg_cosine = total_cosine / len(dataloader)
        history.append({"epoch": epoch + 1, "mse": avg_mse, "cosine": avg_cosine})
        logger.info("epoch=%d  avg_mse=%.4f  avg_cosine=%.4f", epoch + 1, avg_mse, avg_cosine)

    save_payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "preset": args.preset,
            "model_name": cfg.model_name,
            "hidden_size": cfg.hidden_size,
            "small_dim": cfg.small_dim,
            "oracle_type": oracle_type,
            "num_blocks": cfg.num_blocks,
            "cos_weight": cfg.cos_weight,
            "early_count": cfg.early_count,
            "target_layer_start": cfg.target_layer_start,
            "data_dir": data_dir,
            "run_tag": cfg.run_tag,
            "confidence_enabled": confidence_enabled,
        },
        "history": history,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if confidence_enabled:
        save_payload["confidence_head_state_dict"] = confidence_head.state_dict()
        save_payload["config"]["confidence_threshold"] = cfg.confidence_threshold
        save_payload["config"]["confidence_coeff"] = cfg.confidence_coeff
        save_payload["config"]["confidence_head_type"] = confidence_head_type

    torch.save(save_payload, checkpoint)

    with open(Path(checkpoint).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)

    logger.info("checkpoint saved to %s", checkpoint)


if __name__ == "__main__":
    main()
