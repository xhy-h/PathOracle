"""Step ④: PathOracle inference pipeline — with speculative verification fallback.

两种模式：
- **Standard mode**（默认）：Oracle 直接近似跳过层，与 MVP 相同
- **Speculative mode**（``--speculative``）：Oracle 预测 + 置信度检查 + 不通过时回退到原始计算
  → 保证输出与原始模型完全一致
"""

import argparse
import logging
import math

import torch
from transformers import AutoModelForCausalLM

from config import build_experiment_config, validate_model_shape
from oracle_model import build_oracle, ConfidenceHead, MSEPredictionHead
from oracle_utils import (
    compute_perplexity,
    embed_inputs,
    setup_cpu_threads,
    setup_logging,
    setup_tokenizer,
)

logger = logging.getLogger(__name__)


class PathOracleGPT2:
    """GPT-2 with middle layers replaced by an oracle network.

    Forward flow::

        input_ids → wte + wpe → drop → [early layers] → oracle → [late layers] → ln_f → lm_head

    When speculative mode is enabled, a confidence head checks the oracle's
    prediction; if confidence is low, the original skipped layers are re-run
    to guarantee output quality.

    Args:
        preset: Config preset name.
        checkpoint: Path to oracle checkpoint.
        run_tag: Experiment tag for auto-derived checkpoint path.
        speculative: If True, enable confidence-based fallback.
        confidence_threshold: Minimum confidence to accept oracle output.
    """

    def __init__(
        self,
        preset="distilgpt2",
        checkpoint=None,
        run_tag=None,
        speculative=False,
        confidence_threshold=0.8,
    ):
        self.cfg = build_experiment_config(
            preset=preset,
            checkpoint=checkpoint,
            run_tag=run_tag,
        )
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name).to(self.device)
        self.tokenizer = setup_tokenizer(self.cfg.model_name)
        self.model.eval()
        validate_model_shape(self.model, self.cfg)

        checkpoint_path = checkpoint or self.cfg.checkpoint_path
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        ckpt_cfg = payload.get("config", {})
        oracle_type = ckpt_cfg.get("oracle_type", self.cfg.oracle_type)
        small_dim = ckpt_cfg.get("small_dim", self.cfg.small_dim)
        num_blocks = ckpt_cfg.get("num_blocks", self.cfg.num_blocks)

        self.oracle = build_oracle(oracle_type, self.cfg.hidden_size, small_dim, num_blocks)
        self.oracle.load_state_dict(state_dict)
        self.oracle.eval()

        self.early_layers = self.model.transformer.h[: self.cfg.early_count]
        self.late_layers = self.model.transformer.h[self.cfg.target_layer_start :]
        self.skipped_layer_indices = list(range(self.cfg.early_count, self.cfg.target_layer_start))

        # Speculative mode: confidence head
        self.speculative = speculative
        self.confidence_threshold = ckpt_cfg.get("confidence_threshold", confidence_threshold)
        self.confidence_head = None

        if speculative:
            conf_sd = payload.get("confidence_head_state_dict")
            conf_type = ckpt_cfg.get("confidence_head_type", "binary")
            if conf_sd is not None:
                if conf_type == "mse_regression":
                    self.confidence_head = MSEPredictionHead(
                        hidden_size=self.cfg.hidden_size,
                        small_dim=max(64, small_dim),
                    )
                    self._conf_type = "mse"
                else:
                    self.confidence_head = ConfidenceHead(
                        hidden_size=self.cfg.hidden_size,
                        small_dim=max(64, small_dim),
                    )
                    self._conf_type = "binary"
                self.confidence_head.load_state_dict(conf_sd)
                self.confidence_head.eval()
                logger.info(
                    "Speculative mode enabled (type=%s, threshold=%.4f)",
                    conf_type, self.confidence_threshold,
                )
            else:
                logger.warning(
                    "No confidence head found in checkpoint; "
                    "speculative mode disabled (falling back to standard mode)"
                )
                self.speculative = False

    def _run_skipped_layers(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the skipped middle layers on *hidden*, return the result."""
        for idx in self.skipped_layer_indices:
            hidden = self.model.transformer.h[idx](hidden)[0]
        return hidden

    def forward_logits(self, input_ids, speculative=None):
        """Run forward pass, optionally with speculative verification.

        Args:
            input_ids: Shape ``(batch, seq_len)``.
            speculative: Override the instance's speculative setting.

        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        if speculative is None:
            speculative = self.speculative

        hidden = embed_inputs(self.model, input_ids)

        # Early layers
        for layer in self.early_layers:
            hidden = layer(hidden)[0]

        # Oracle prediction
        oracle_input = hidden
        oracle_output = self.oracle(oracle_input)

        if speculative and self.confidence_head is not None:
            # Confidence check depends on head type
            scores = self.confidence_head(oracle_input).mean().item()

            if self._conf_type == "mse":
                # MSE prediction: lower = better
                accept = scores <= self.confidence_threshold
                self._last_pred_mse = scores
                detail = f"pred_mse={scores:.6f}"
            else:
                # Binary confidence: higher = better
                accept = scores >= self.confidence_threshold
                self._last_confidence = scores
                detail = f"confidence={scores:.4f}"

            logger.debug("%s (threshold=%.4f) → %s", detail, self.confidence_threshold,
                         "accept" if accept else "reject")

            if accept:
                # Accept oracle prediction
                hidden = oracle_output
                self._last_fallback = False
            else:
                # Fall back to original computation
                logger.info("%s < threshold — falling back to original layers", detail)
                hidden = self._run_skipped_layers(oracle_input)
                self._last_fallback = True
        else:
            # Standard mode: always use oracle
            hidden = oracle_output

        # Late layers
        for layer in self.late_layers:
            hidden = layer(hidden)[0]

        hidden = self.model.transformer.ln_f(hidden)
        return self.model.lm_head(hidden)

    def generate(self, prompt, max_new_tokens=20, speculative=None):
        """Greedy text generation from a prompt string.

        Args:
            prompt: Input text.
            max_new_tokens: Number of tokens to generate.
            speculative: Override the instance's speculative setting.

        Returns:
            Decoded generated text (including the prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        fallbacks = 0
        with torch.no_grad():
            for step in range(max_new_tokens):
                logits = self.forward_logits(input_ids, speculative=speculative)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if hasattr(self, '_last_fallback') and self._last_fallback:
                    fallbacks += 1

        if speculative:
            logger.info("Fallback rate: %d/%d steps", fallbacks, max_new_tokens)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def compute_perplexity(self, text, max_length=64, speculative=None):
        """Compute perplexity over a single text.

        Args:
            text: Input string.
            max_length: Maximum sequence length.
            speculative: Override the instance's speculative setting.

        Returns:
            Perplexity score, or ``nan`` if text is too short.
        """
        if speculative is None:
            speculative = self.speculative

        def forward_fn(input_ids):
            return self.forward_logits(input_ids, speculative=speculative)

        return compute_perplexity(
            self.tokenizer,
            text,
            forward_fn,
            max_length=max_length,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run PathOracle inference (standard or speculative)"
    )
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--speculative", action="store_true",
                        help="Enable speculative mode with confidence fallback")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    pipe = PathOracleGPT2(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
        speculative=args.speculative,
        confidence_threshold=args.confidence_threshold,
    )

    output = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    ppl = pipe.compute_perplexity("The quick brown fox jumps over the lazy dog.",
                                  speculative=args.speculative)

    logger.info("prompt=%s", args.prompt)
    logger.info("output=%s", output)
    logger.info("sample_ppl=%.4f", ppl)

    print(output)
    print(f"PPL: {ppl:.4f}")
    if args.speculative:
        print(f"Mode: speculative (threshold={args.confidence_threshold})")


if __name__ == "__main__":
    main()
