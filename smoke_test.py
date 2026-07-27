"""Step ①: smoke test — validate model shape and estimate oracle parameter count."""

import argparse
import logging

from transformers import AutoModelForCausalLM

from config import build_experiment_config, validate_model_shape
from oracle_model import build_oracle, count_parameters
from oracle_utils import setup_cpu_threads, setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smoke test / environment validation")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--oracle-type", default=None, choices=["mlp", "transformer"])
    parser.add_argument("--small-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(
        preset=args.preset,
        oracle_type=args.oracle_type,
        small_dim=args.small_dim,
        num_blocks=args.num_blocks,
    )
    oracle_type = cfg.oracle_type

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    validate_model_shape(model, cfg)

    oracle = build_oracle(oracle_type, cfg.hidden_size, cfg.small_dim, cfg.num_blocks)

    logger.info("model=%s", cfg.model_name)
    logger.info("layers=%d", len(model.transformer.h))
    logger.info("hidden_size=%d", model.config.n_embd)
    logger.info("early_layers=0..%d", cfg.early_count - 1)
    logger.info("skipped_layers=%d..%d", cfg.early_count, cfg.target_layer_start - 1)
    logger.info("late_layers=%d..%d", cfg.target_layer_start, cfg.total_layers - 1)
    logger.info("oracle_type=%s", oracle_type)
    logger.info("small_dim=%d", cfg.small_dim)
    logger.info("num_blocks=%d", cfg.num_blocks)
    logger.info("oracle_parameters=%d", count_parameters(oracle))


if __name__ == "__main__":
    main()
