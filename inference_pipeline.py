import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import build_experiment_config, validate_model_shape
from oracle_model import build_oracle


class PathOracleGPT2:
    def __init__(self, preset="distilgpt2", checkpoint=None, run_tag=None):
        self.cfg = build_experiment_config(
            preset=preset,
            checkpoint=checkpoint,
            run_tag=run_tag,
        )
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        validate_model_shape(self.model, self.cfg)

        checkpoint_path = checkpoint or self.cfg.checkpoint_path
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        ckpt_cfg = payload.get("config", {})
        oracle_type = ckpt_cfg.get("oracle_type", self.cfg.oracle_type)
        small_dim = ckpt_cfg.get("small_dim", self.cfg.small_dim)
        num_blocks = ckpt_cfg.get("num_blocks", self.cfg.num_blocks)

        self.oracle = build_oracle(
            oracle_type,
            self.cfg.hidden_size,
            small_dim,
            num_blocks,
        )
        self.oracle.load_state_dict(state_dict)
        self.oracle.eval()

        self.early_layers = self.model.transformer.h[: self.cfg.early_count]
        self.late_layers = self.model.transformer.h[self.cfg.target_layer_start :]

    def _embed(self, input_ids):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.model.transformer.wte(input_ids) + self.model.transformer.wpe(position_ids)
        return self.model.transformer.drop(hidden)

    def forward_logits(self, input_ids):
        hidden = self._embed(input_ids)
        for layer in self.early_layers:
            hidden = layer(hidden)[0]
        hidden = self.oracle(hidden)
        for layer in self.late_layers:
            hidden = layer(hidden)[0]
        hidden = self.model.transformer.ln_f(hidden)
        return self.model.lm_head(hidden)

    def generate(self, prompt, max_new_tokens=20):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward_logits(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def compute_perplexity(self, text, max_length=64):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        if input_ids.size(1) < 2:
            return math.nan

        with torch.no_grad():
            logits = self.forward_logits(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="distilgpt2", choices=["distilgpt2", "gpt2"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    pipe = PathOracleGPT2(
        preset=args.preset,
        checkpoint=args.checkpoint,
        run_tag=args.run_tag,
    )
    output = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    ppl = pipe.compute_perplexity("The quick brown fox jumps over the lazy dog.")
    print(f"prompt={args.prompt}")
    print(f"output={output}")
    print(f"sample_ppl={ppl:.4f}")


if __name__ == "__main__":
    main()
