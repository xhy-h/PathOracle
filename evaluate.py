"""Step ⑤: evaluate PathOracle — generation comparison + WikiText-2 PPL."""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from config import build_experiment_config
from inference_pipeline import PathOracleGPT2
from oracle_utils import setup_cpu_threads, setup_logging, setup_tokenizer

logger = logging.getLogger(__name__)


PROMPTS = [
    "The weather today is",
    "My favorite book is",
    "In the year 2050,",
    "The recipe for cake",
    "Once upon a time",
    "The theory of relativity",
    "Python programming language",
    "The capital of Germany is",
    "To improve health, one should",
    "Artificial intelligence will",
    "The stock market",
    "A good night's sleep",
    "The best way to learn",
    "Climate change is",
    "The history of the internet",
    "In a small village,",
    "The function of the heart is to",
    "The novel begins with",
    "The future of space travel",
    "The main character of the story",
]


def compute_original_ppl(model, tokenizer, text, max_length):
    """Compute perplexity of the original (unmodified) model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs["input_ids"]
    if input_ids.size(1) < 2:
        return math.nan
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PathOracle vs original model")
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--max-texts", type=int, default=50)
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--speculative", action="store_true",
                        help="Use speculative inference with confidence fallback")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_cpu_threads()
    setup_logging(verbose=args.verbose)

    cfg = build_experiment_config(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
    )
    tokenizer = setup_tokenizer(cfg.model_name)
    original = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    original.eval()
    pathoracle = PathOracleGPT2(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
        speculative=args.speculative,
        confidence_threshold=args.confidence_threshold,
    )

    # ── Generation comparison ──
    logger.info("=== generation comparison ===")
    gen_results = []
    for prompt in PROMPTS[: args.max_prompts]:
        encoded = tokenizer(prompt, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            original_ids = original.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        original_time = time.perf_counter() - start

        start = time.perf_counter()
        path_text = pathoracle.generate(prompt, max_new_tokens=args.max_new_tokens)
        path_time = time.perf_counter() - start

        original_text = tokenizer.decode(original_ids, skip_special_tokens=True)
        logger.info("prompt: %s", prompt)
        logger.info("  original (%.2fs): %s", original_time, original_text)
        logger.info("  pathoracle (%.2fs): %s", path_time, path_text)

        gen_results.append({
            "prompt": prompt,
            "original_text": original_text,
            "original_time_s": original_time,
            "pathoracle_text": path_text,
            "pathoracle_time_s": path_time,
        })

    # ── Perplexity comparison ──
    logger.info("=== perplexity comparison ===")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [
        item["text"]
        for item in dataset
        if len(item["text"].strip()) > 50
    ][: args.max_texts]

    original_ppls = []
    path_ppls = []
    for text in texts:
        original_ppl = compute_original_ppl(original, tokenizer, text, cfg.max_length)
        path_ppl = pathoracle.compute_perplexity(text, max_length=cfg.max_length,
                                                   speculative=args.speculative)
        if not math.isnan(original_ppl) and not math.isnan(path_ppl):
            original_ppls.append(original_ppl)
            path_ppls.append(path_ppl)

    avg_original = sum(original_ppls) / len(original_ppls)
    avg_path = sum(path_ppls) / len(path_ppls)
    relative_increase = (avg_path / avg_original - 1.0) * 100.0

    logger.info("texts=%d", len(original_ppls))
    logger.info("original_ppl=%.4f  pathoracle_ppl=%.4f  relative_increase=%.2f%%",
                avg_original, avg_path, relative_increase)

    # ── Save structured results ──
    results = {
        "config": {
            "preset": args.preset,
            "run_tag": args.run_tag,
            "max_texts": args.max_texts,
            "max_prompts": args.max_prompts,
            "max_new_tokens": args.max_new_tokens,
            "speculative": args.speculative,
            "confidence_threshold": args.confidence_threshold if args.speculative else None,
        },
        "generation": gen_results,
        "perplexity": {
            "num_texts": len(original_ppls),
            "original_ppl": round(avg_original, 4),
            "pathoracle_ppl": round(avg_path, 4),
            "relative_increase_pct": round(relative_increase, 2),
        },
    }

    output_path = args.output
    if not output_path:
        tag = args.run_tag or cfg.model_short
        output_path = f"eval_results_{tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Console summary
    print(f"Original PPL:  {avg_original:.4f}")
    print(f"PathOracle PPL: {avg_path:.4f}")
    print(f"Relative increase: {relative_increase:.2f}%")


if __name__ == "__main__":
    main()
