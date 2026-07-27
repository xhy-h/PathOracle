"""Tests for oracle_utils shared functions."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from oracle_utils import compute_perplexity, setup_cpu_threads, setup_tokenizer


class SetupCpuThreadsTests(unittest.TestCase):
    def test_sets_nonzero_threads(self):
        setup_cpu_threads()
        self.assertGreater(torch.get_num_threads(), 0)


class SetupTokenizerTests(unittest.TestCase):
    def test_pad_token_matches_eos(self):
        tokenizer = setup_tokenizer("distilgpt2")
        self.assertIsNotNone(tokenizer.pad_token)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)


class ComputePerplexityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = setup_tokenizer("distilgpt2")

    def _identity_logits(self, input_ids):
        """Return dummy logits: random normal (won't produce sensible PPL but shape is right)."""
        batch, seq = input_ids.shape
        vocab = self.tokenizer.vocab_size
        return torch.randn(batch, seq, vocab)

    def test_returns_positive_number_for_long_enough_text(self):
        ppl = compute_perplexity(
            self.tokenizer,
            "The quick brown fox jumps over the lazy dog.",
            self._identity_logits,
            max_length=64,
        )
        self.assertIsNotNone(ppl)
        self.assertNotEqual(ppl, math.nan)
        self.assertGreater(ppl, 0.0)

    def test_returns_nan_for_single_token(self):
        ppl = compute_perplexity(
            self.tokenizer,
            "Hello",
            self._identity_logits,
            max_length=64,
        )
        self.assertTrue(math.isnan(ppl))

    def test_truncates_to_max_length(self):
        long_text = "word " * 100
        ppl = compute_perplexity(
            self.tokenizer,
            long_text,
            self._identity_logits,
            max_length=16,
        )
        self.assertFalse(math.isnan(ppl))


if __name__ == "__main__":
    unittest.main()
