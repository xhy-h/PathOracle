"""Tests for PathOracleGPT2 inference pipeline (lightweight, no full model load)."""

import unittest
from unittest.mock import patch

from config import build_experiment_config, get_config, validate_model_shape


class ConfigLayerSliceTests(unittest.TestCase):
    """Verify that config-derived layer indices are self-consistent."""

    def test_distilgpt2_layer_bounds(self):
        cfg = get_config("distilgpt2")
        # early + skipped + late should cover all layers
        early = cfg.early_count
        skipped = cfg.target_layer_start - cfg.early_count
        late = cfg.late_count
        self.assertEqual(early + skipped + late, cfg.total_layers)

    def test_gpt2_layer_bounds(self):
        cfg = get_config("gpt2")
        early = cfg.early_count
        skipped = cfg.target_layer_start - cfg.early_count
        late = cfg.late_count
        self.assertEqual(early + skipped + late, cfg.total_layers)

    def test_validate_model_shape_passes_for_valid_config(self):
        """Create a mock model that satisfies distilgpt2's expected shape."""
        cfg = get_config("distilgpt2")

        class MockConfig:
            n_embd = cfg.hidden_size

        class MockTransformer:
            h = list(range(cfg.total_layers))

        class MockModel:
            config = MockConfig()
            transformer = MockTransformer()

        # Should not raise
        try:
            validate_model_shape(MockModel(), cfg)
        except ValueError as exc:
            self.fail(f"validate_model_shape raised unexpectedly: {exc}")

    def test_validate_model_shape_raises_on_layer_mismatch(self):
        cfg = build_experiment_config(preset="distilgpt2")

        class MockConfig:
            n_embd = cfg.hidden_size

        class MockTransformer:
            h = list(range(cfg.total_layers + 1))  # one extra layer

        class MockModel:
            config = MockConfig()
            transformer = MockTransformer()

        with self.assertRaises(ValueError):
            validate_model_shape(MockModel(), cfg)

    def test_validate_model_shape_raises_on_bad_target_start(self):
        """target_layer_start must be > early_count."""
        cfg = build_experiment_config(preset="distilgpt2")
        # Override to violate invariant
        invalid_cfg = build_experiment_config(
            preset="distilgpt2",
            run_tag="test_bad_target",
        )
        # Manually create a bad config
        from dataclasses import replace
        bad_cfg = replace(invalid_cfg, early_count=3, target_layer_start=2)

        class MockConfig:
            n_embd = bad_cfg.hidden_size

        class MockTransformer:
            h = list(range(bad_cfg.total_layers))

        class MockModel:
            config = MockConfig()
            transformer = MockTransformer()

        with self.assertRaises(ValueError):
            validate_model_shape(MockModel(), bad_cfg)


if __name__ == "__main__":
    unittest.main()
