import argparse

from transformers import AutoModelForCausalLM

from config import build_experiment_config, validate_model_shape
from oracle_model import build_oracle, count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--oracle-type", default=None, choices=["mlp", "transformer"])
    parser.add_argument("--small-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    args = parser.parse_args()

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
    print(f"model={cfg.model_name}")
    print(f"layers={len(model.transformer.h)}")
    print(f"hidden_size={model.config.n_embd}")
    print(f"early_layers=0..{cfg.early_count - 1}")
    print(f"skipped_layers={cfg.early_count}..{cfg.target_layer_start - 1}")
    print(f"late_layers={cfg.target_layer_start}..{cfg.total_layers - 1}")
    print(f"oracle_type={oracle_type}")
    print(f"small_dim={cfg.small_dim}")
    print(f"num_blocks={cfg.num_blocks}")
    print(f"oracle_parameters={count_parameters(oracle)}")


if __name__ == "__main__":
    main()
