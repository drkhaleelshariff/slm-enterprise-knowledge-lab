import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "command_assistant_v3"


def main():
    torch.set_num_threads(2)

    training_report = json.loads(
        (ROOT / "results" / "v3_lora_training.json").read_text(
            encoding="utf-8"
        )
    )

    # Ensure evaluation uses the same shared instruction as training.
    instruction_path = DATA_DIR / "instruction.txt"

    current_hash = hashlib.sha256(
        instruction_path.read_bytes()
    ).hexdigest()

    if current_hash != training_report["instruction_file_sha256"]:
        raise ValueError("Instruction changed after training")

    shared_instruction = training_report["shared_instruction"]

    development_path = DATA_DIR / "development.jsonl"

    cases = [
        json.loads(line)
        for line in development_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    model_id = training_report["model_id"]
    revision = training_report["model_revision"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model and V3 adapter...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model = PeftModel.from_pretrained(
        base_model,
        ROOT / "adapters" / "smollm2-command-v3",
        is_trainable=False,
    )

    model.eval()

    settings = {
        "max_new_tokens": 32,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    records = []

    for case in cases:
        prompt = (
            shared_instruction
            + "\n\n"
            + f"Question: {case['instruction']}\n"
        )

        if case["input"]:
            prompt += f"Context: {case['input']}\n"

        prompt += "Answer:\n"

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_length = inputs["input_ids"].shape[1]

        print(f"\n{case['id']} | {case['evaluation_type']}")
        print("Question:", case["instruction"])
        print("Expected:", case["output"])

        for condition in ("base", "v3_adapter"):
            context = (
                model.disable_adapter()
                if condition == "base"
                else nullcontext()
            )

            with context:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        **settings,
                    )

            new_ids = output[0, prompt_length:]

            answer = tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            ).strip()

            exact_match = answer == case["output"].strip()

            ended_with_eos = (
                new_ids.numel() > 0
                and new_ids[-1].item() == tokenizer.eos_token_id
            )

            records.append({
                "id": case["id"],
                "evaluation_type": case["evaluation_type"],
                "condition": condition,
                "prompt": prompt,
                "expected": case["output"],
                "answer": answer,
                "exact_match": exact_match,
                "ended_with_eos": ended_with_eos,
                "generated_tokens": new_ids.numel(),
                "human_review": "pending",
            })

            print(f"  {condition}: {answer!r}")
            print(
                f"  Exact match: {exact_match} | "
                f"Ended with EOS: {ended_with_eos}",
                flush=True,
            )

    # Report categories separately rather than hiding them in one score.
    summaries = {}

    for condition in ("base", "v3_adapter"):
        summaries[condition] = {}

        for category in sorted({
            case["evaluation_type"] for case in cases
        }):
            selected = [
                record for record in records
                if record["condition"] == condition
                and record["evaluation_type"] == category
            ]

            summaries[condition][category] = {
                "exact_matches": sum(
                    record["exact_match"] for record in selected
                ),
                "total": len(selected),
            }

    report = {
        "model_id": model_id,
        "model_revision": revision,
        "development_file_sha256": hashlib.sha256(
            development_path.read_bytes()
        ).hexdigest(),
        "generation_settings": settings,
        "summary": summaries,
        "records": records,
        "limitations": [
            "Six visible development questions, not a sealed test set.",
            "Exact matching can reject valid alternative commands or wording.",
            "One unknown-fact probe does not establish reliable abstention.",
            "V1 and V2 differ in data and instructions; effects are not isolated.",
        ],
    }

    output_path = ROOT / "results" / "v3_development_evaluation.json"

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nExact-match summary:")
    print(json.dumps(summaries, indent=2))
    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()
