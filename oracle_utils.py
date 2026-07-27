"""Shared utilities for the PathOracle CPU MVP pipeline."""

import logging
import math
import os
from typing import Callable

import torch
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def setup_cpu_threads() -> None:
    """Configure PyTorch CPU thread count from ``OMP_NUM_THREADS`` (default 4)."""
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(threads)
    logger.info("CPU threads set to %d", threads)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with a simple console handler.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer and set ``pad_token = eos_token``."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def embed_inputs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """GPT-2 embedding lookup: ``wte + wpe + dropout``.

    Args:
        model: A HuggingFace GPT-2 (or compatible) model.
        input_ids: Shape ``(batch, seq_len)``.

    Returns:
        Embedded hidden states, shape ``(batch, seq_len, hidden_size)``.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = (
        torch.arange(seq_len, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(position_ids)
    return model.transformer.drop(hidden)


def compute_perplexity(
    tokenizer: AutoTokenizer,
    text: str,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    max_length: int = 64,
) -> float:
    """Compute perplexity over a single text via a user-supplied forward function.

    Args:
        tokenizer: Tokenizer for encoding the text.
        text: Input string.
        forward_fn: A callable ``(input_ids) -> logits`` of shape ``(batch, seq, vocab)``.
        max_length: Maximum sequence length (truncation).

    Returns:
        Perplexity (``exp(loss)``), or ``nan`` if the text is too short.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    if input_ids.size(1) < 2:
        return math.nan

    with torch.no_grad():
        logits = forward_fn(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return torch.exp(loss).item()
