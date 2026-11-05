import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)

    # Resolve a commit once, then use it for both conditions.
    # Keep this small configuration file in Git for future reruns.
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    revision_path = config_dir / "model_revision.json"

    if revision_path.exists():
        revision_config = json.loads(
            revision_path.read_text(encoding="utf-8")
        )
        if revision_config["model_id"] != MODEL_ID:
            raise ValueError("Saved revision belongs to another model")

        revision = revision_config["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha

        if not revision:
            raise ValueError("Could not resolve model revision")

        revision_path.write_text(
            json.dumps(
                {"model_id": MODEL_ID, "revision": revision},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    print("Model:", MODEL_ID)
    print("Pinned revision:", revision)
    print("Loading on CPU...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model.eval()

    instructions = (
        "Answer IT support questions briefly and accurately.\n"
        "For command questions, give the command only.\n"
        "For company-specific facts without supplied evidence, "
        "answer: I do not know from the supplied information.\n\n"
    )

    demonstrations = (
        "Question: Which Linux command prints the current directory?\n"
        "Answer: pwd\n\n"
        "Question: Which Linux command displays the current user name?\n"
        "Answer: whoami\n\n"
        "Question: What is the VPN password for fictional Cedar Works?\n"
        "Answer: I do not know from the supplied information.\n\n"
    )

    cases = [
        {
            "id": "linux-disk",
            "question": "Which Linux command displays filesystem space?",
            "expected": "df or df -h",
        },
        {
            "id": "python-version",
            "question": "Which command displays the installed Python version?",
            "expected": "python --version, python -V, or equivalent",
        },
        {
            "id": "unknown-policy",
            "question": (
                "What is the incident escalation phone number "
                "at fictional Meridian Labs?"
            ),
            "expected": (
                "Acknowledge that the supplied information "
                "does not contain the number."
            ),
        },
    ]

    settings = {
        "max_new_tokens": 40,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Exclude initial execution setup from the measured cases.
    warmup = tokenizer("Hello", return_tensors="pt")

    with torch.inference_mode():
        model.generate(
            **warmup,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    records = []

    for case in cases:
        for condition in ("zero_shot", "few_shot"):
            examples = (
                demonstrations if condition == "few_shot" else ""
            )

            prompt = (
                instructions
                + examples
                + f"Question: {case['question']}\nAnswer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = inputs["input_ids"].shape[1]

            start = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(**inputs, **settings)

            elapsed = time.perf_counter() - start

            new_ids = output[0, input_length:]

            continuation = tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            )

            # Preserve the full continuation for review.
            # Do not hide additional or contradictory text.
            record = {
                "case_id": case["id"],
                "condition": condition,
                "prompt": prompt,
                "continuation": continuation,
                "expected_behaviour": case["expected"],
                "input_tokens": input_length,
                "generated_tokens": new_ids.numel(),
                "generation_seconds": elapsed,
                "human_review": "pending",
            }

            records.append(record)

            print(f"\n{case['id']} | {condition}")
            print("Input tokens:", input_length)
            print("Continuation:", repr(continuation))
            print(f"Generation seconds: {elapsed:.3f}")

    results = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": torch.get_num_threads(),
        "generation_settings": settings,
        "evaluation_scope": (
            "Three exploratory cases, two prompt conditions. "
            "No training or automatic correctness scoring."
        ),
        "records": records,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "prompt_comparison.json"

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()