import argparse
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import build_experiment_config
from inference_pipeline import PathOracleGPT2


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
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs["input_ids"]
    if input_ids.size(1) < 2:
        return math.nan
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--max-texts", type=int, default=50)
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    cfg = build_experiment_config(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    original = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    original.eval()
    pathoracle = PathOracleGPT2(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
    )

    print("=== generation comparison ===")
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
        print(f"prompt: {prompt}")
        print(f"original ({original_time:.2f}s): {original_text}")
        print(f"pathoracle ({path_time:.2f}s): {path_text}")
        print("---")

    print("=== perplexity comparison ===")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][
        : args.max_texts
    ]

    original_ppls = []
    path_ppls = []
    for text in texts:
        original_ppl = compute_original_ppl(original, tokenizer, text, cfg.max_length)
        path_ppl = pathoracle.compute_perplexity(text, max_length=cfg.max_length)
        if not math.isnan(original_ppl) and not math.isnan(path_ppl):
            original_ppls.append(original_ppl)
            path_ppls.append(path_ppl)

    avg_original = sum(original_ppls) / len(original_ppls)
    avg_path = sum(path_ppls) / len(path_ppls)
    print(f"texts={len(original_ppls)}")
    print(f"original_ppl={avg_original:.4f}")
    print(f"pathoracle_ppl={avg_path:.4f}")
    print(f"relative_increase={(avg_path / avg_original - 1.0) * 100.0:.2f}%")


if __name__ == "__main__":
    main()
