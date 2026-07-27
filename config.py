from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    model_short: str
    hidden_size: int
    total_layers: int
    early_count: int
    late_count: int
    target_layer_start: int
    small_dim: int
    max_length: int
    collect_samples: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    oracle_type: str
    num_blocks: int
    cos_weight: float
    dataset_name: str
    dataset_config: Optional[str]
    dataset_split: str
    streaming: bool
    run_tag: str
    data_dir: str
    checkpoint_path: str
    # Plan B: Speculative PathOracle
    confidence_threshold: float = 0.8
    confidence_coeff: float = 0.1


PRESETS = {
    "distilgpt2": ExperimentConfig(
        model_name="distilgpt2",
        model_short="distilgpt2",
        hidden_size=768,
        total_layers=6,
        early_count=2,
        late_count=2,
        target_layer_start=4,
        small_dim=96,
        max_length=64,
        collect_samples=512,
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-3,
        oracle_type="mlp",
        num_blocks=1,
        cos_weight=0.1,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="train",
        streaming=False,
        run_tag="",
        data_dir="oracle_data_distilgpt2",
        checkpoint_path="oracle_distilgpt2.pth",
    ),
    "gpt2": ExperimentConfig(
        model_name="gpt2",
        model_short="gpt2",
        hidden_size=768,
        total_layers=12,
        early_count=2,
        late_count=2,
        target_layer_start=10,
        small_dim=128,
        max_length=64,
        collect_samples=2000,
        batch_size=2,
        num_epochs=3,
        learning_rate=1e-3,
        oracle_type="mlp",
        num_blocks=1,
        cos_weight=0.1,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="train",
        streaming=False,
        run_tag="",
        data_dir="oracle_data_gpt2",
        checkpoint_path="oracle_gpt2.pth",
    ),
}


def get_config(preset: str) -> ExperimentConfig:
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}")
    return PRESETS[preset]


def resolve_dataset(dataset: Optional[str], dataset_config: Optional[str]):
    if dataset in (None, "", "wikitext", "wikitext-2-raw-v1"):
        return "wikitext", dataset_config or "wikitext-2-raw-v1"
    if dataset == "c4":
        return "c4", dataset_config or "en"
    return dataset, dataset_config


def build_experiment_config(
    preset: str = "distilgpt2",
    samples: Optional[int] = None,
    max_length: Optional[int] = None,
    small_dim: Optional[int] = None,
    oracle_type: Optional[str] = None,
    num_blocks: Optional[int] = None,
    cos_weight: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    dataset: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: Optional[str] = None,
    streaming: Optional[bool] = None,
    run_tag: Optional[str] = None,
    data_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> ExperimentConfig:
    cfg = get_config(preset)
    resolved_dataset, resolved_dataset_config = resolve_dataset(dataset, dataset_config)
    tag = run_tag or cfg.run_tag

    cfg = replace(
        cfg,
        collect_samples=samples if samples is not None else cfg.collect_samples,
        max_length=max_length if max_length is not None else cfg.max_length,
        small_dim=small_dim if small_dim is not None else cfg.small_dim,
        oracle_type=oracle_type or cfg.oracle_type,
        num_blocks=num_blocks if num_blocks is not None else cfg.num_blocks,
        cos_weight=cos_weight if cos_weight is not None else cfg.cos_weight,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        num_epochs=epochs if epochs is not None else cfg.num_epochs,
        learning_rate=lr if lr is not None else cfg.learning_rate,
        dataset_name=resolved_dataset,
        dataset_config=resolved_dataset_config,
        dataset_split=dataset_split or cfg.dataset_split,
        streaming=streaming if streaming is not None else cfg.streaming,
        run_tag=tag,
    )

    if tag:
        cfg = replace(
            cfg,
            data_dir=data_dir or f"oracle_data_{tag}",
            checkpoint_path=checkpoint or f"oracle_{tag}.pth",
        )
    else:
        cfg = replace(
            cfg,
            data_dir=data_dir or cfg.data_dir,
            checkpoint_path=checkpoint or cfg.checkpoint_path,
        )

    return cfg


def validate_model_shape(model, cfg: ExperimentConfig) -> None:
    actual_layers = len(model.transformer.h)
    actual_hidden = model.config.n_embd
    if actual_layers != cfg.total_layers:
        raise ValueError(
            f"{cfg.model_name} layer mismatch: expected {cfg.total_layers}, got {actual_layers}"
        )
    if actual_hidden != cfg.hidden_size:
        raise ValueError(
            f"{cfg.model_name} hidden size mismatch: expected {cfg.hidden_size}, got {actual_hidden}"
        )
    if cfg.target_layer_start <= cfg.early_count:
        raise ValueError("target_layer_start must be greater than early_count")
    if cfg.target_layer_start + cfg.late_count != cfg.total_layers:
        raise ValueError("target_layer_start must mark the first late layer")
