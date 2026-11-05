import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)

    training = json.loads(
        (ROOT / "results" / "v2_lora_training.json").read_text(
            encoding="utf-8"
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training["model_id"],
        revision=training["model_revision"],
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        training["model_id"],
        revision=training["model_revision"],
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model = PeftModel.from_pretrained(
        base,
        ROOT / "adapters" / "smollm2-command-v2",
        is_trainable=False,
    )
    model.eval()

    cases = [
        {
            "category": "training_question",
            "question": "Which command displays the installed Python version?",
        },
        {
            "category": "training_question",
            "question": (
                "Print the version of the Python interpreter invoked as python."
            ),
        },
        {
            "category": "development_question",
            "question": (
                "How can I ask the python executable to report its version and exit?"
            ),
        },
    ]

    records = []

    for case in cases:
        prompt = (
            training["shared_instruction"]
            + "\n\n"
            + f"Question: {case['question']}\n"
            + "Answer:\n"
        )

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output[0, inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()

        record = {
            **case,
            "answer": answer,
            "exact_training_target_match": answer == "python --version",
            "accepted_command_match": answer in {
                "python --version",
                "python -V",
            },
            "ended_with_eos": (
                new_ids.numel() > 0
                and new_ids[-1].item() == tokenizer.eos_token_id
            ),
        }

        records.append(record)

        print(f"\nCategory: {case['category']}")
        print("Question:", case["question"])
        print("Answer:", repr(answer))
        print("Exact target match:", record["exact_training_target_match"])
        print("Accepted command:", record["accepted_command_match"])

    report = {
        "model_id": training["model_id"],
        "model_revision": training["model_revision"],
        "purpose": "Diagnose Python-version failure without updating weights",
        "records": records,
    }

    output_path = ROOT / "results" / "v2_python_diagnostic.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()