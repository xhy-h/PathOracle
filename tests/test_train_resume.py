import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from train_oracle import load_resume_checkpoint, resolve_resume_checkpoint


class TrainResumeTests(unittest.TestCase):
    def test_resolve_resume_from_accepts_run_tag(self):
        path = resolve_resume_checkpoint(
            checkpoint_path="oracle_current.pth",
            resume=False,
            resume_from="distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3",
        )

        self.assertEqual(
            path,
            Path("oracle_distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3.pth"),
        )

    def test_load_resume_checkpoint_uses_existing_history_without_optimizer_state(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "oracle_old.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": [
                        {"epoch": 1, "mse": 3.0, "cosine": 0.7},
                        {"epoch": 2, "mse": 2.0, "cosine": 0.8},
                    ],
                },
                checkpoint,
            )

            history, start_epoch = load_resume_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
            )

        self.assertEqual(start_epoch, 2)
        self.assertEqual(len(history), 2)

    def test_load_resume_checkpoint_can_reset_optimizer_for_finetuning(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "oracle_old.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": [{"epoch": 1, "mse": 3.0, "cosine": 0.7}],
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint,
            )

            fine_tune_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            fine_tune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                fine_tune_optimizer,
                T_max=8,
            )
            history, start_epoch = load_resume_checkpoint(
                checkpoint,
                model,
                fine_tune_optimizer,
                fine_tune_scheduler,
                reset_optimizer=True,
            )

        self.assertEqual(start_epoch, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(fine_tune_optimizer.param_groups[0]["lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
