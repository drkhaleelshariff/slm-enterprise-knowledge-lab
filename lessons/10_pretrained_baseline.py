import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)

    print("Loading tokenizer and model on CPU...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model.eval()

    load_seconds = time.perf_counter() - load_start

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    print("Model:", MODEL_ID)
    print("Parameters:", f"{parameter_count:,}")
    print("Parameter storage MiB:", round(parameter_bytes / 1024**2, 2))
    print("Load time seconds:", round(load_seconds, 2))

    generation_settings = {
        "max_new_tokens": 40,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Warm up execution before measuring the examples.
    warmup = tokenizer("Hello", return_tensors="pt")

    with torch.inference_mode():
        model.generate(
            **warmup,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    cases = [
        {
            "id": "linux-completion",
            "prompt": "To check disk space on Linux, run",
            "review_hint": "Does the continuation suggest df or df -h?",
        },
        {
            "id": "python-completion",
            "prompt": "To display the installed Python version, run",
            "review_hint": "Does it suggest python --version or equivalent?",
        },
        {
            "id": "unknown-policy",
            "prompt": (
                "Question: What is the incident escalation phone number "
                "at the fictional company Meridian Labs?\nAnswer:"
            ),
            "review_hint": (
                "No policy evidence was supplied. Does it invent details "
                "or acknowledge that the number is unknown?"
            ),
        },
    ]

    records = []

    for case in cases:
        inputs = tokenizer(
            case["prompt"],
            return_tensors="pt",
        )

        input_length = inputs["input_ids"].shape[1]

        start = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **generation_settings,
            )

        elapsed = time.perf_counter() - start

        # generate() returns the prompt followed by new tokens.
        new_ids = output_ids[0, input_length:]
        generated_count = new_ids.numel()

        continuation = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )

        record = {
            "id": case["id"],
            "prompt": case["prompt"],
            "continuation": continuation,
            "input_tokens": input_length,
            "generated_tokens_including_special": generated_count,
            "generation_seconds": elapsed,
            "generated_tokens_per_second": generated_count / elapsed,
            "review_hint": case["review_hint"],
            "human_review": "pending",
        }

        records.append(record)

        print(f"\nCase: {case['id']}")
        print("Prompt:", case["prompt"])
        print("Continuation:", repr(continuation))
        print("Generated tokens:", generated_count)
        print(f"Generation seconds: {elapsed:.3f}")
        print(
            "Generated tokens/second:",
            f"{generated_count / elapsed:.2f}",
        )
        print("Review:", case["review_hint"])

    results = {
        "model_id": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "evaluation_scope": "three illustrative prompts; not a benchmark",
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "parameter_count": parameter_count,
        "parameter_storage_bytes": parameter_bytes,
        "model_load_seconds_including_download_if_needed": load_seconds,
        "generation_settings": generation_settings,
        "cases": records,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "pretrained_baseline.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()