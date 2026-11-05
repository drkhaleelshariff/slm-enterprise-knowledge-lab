import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_num_threads(2)

    training_report = json.loads(
        (ROOT / "results" / "lora_training.json").read_text(
            encoding="utf-8"
        )
    )

    prepared = json.loads(
        (ROOT / "results" / "sft_labels.json").read_text(
            encoding="utf-8"
        )
    )

    model_id = training_report["model_id"]
    revision = training_report["model_revision"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model = PeftModel.from_pretrained(
        base_model,
        ROOT / "adapters" / "smollm2-command-demo",
        is_trainable=False,
    )

    model.eval()

    batch = {
        name: torch.tensor(prepared[name], dtype=torch.long)
        for name in ("input_ids", "attention_mask", "labels")
    }

    # Verify the saved adapter before evaluating generation.
    with torch.inference_mode():
        restored_loss = model(
            **batch,
            use_cache=False,
        ).loss.item()

    saved_loss = training_report["final_answer_loss"]
    loss_matches = abs(restored_loss - saved_loss) < 1e-5

    print(f"Saved training loss:    {saved_loss:.8f}")
    print(f"Restored training loss: {restored_loss:.8f}")
    print("Reload check passed:", loss_matches)

    assert loss_matches, "Restored adapter loss differs from saved result"

    cases = [
        {
            "id": "seen-pwd",
            "category": "training_question",
            "question": (
                "Which Linux command prints the working directory?"
            ),
            "expected": "pwd",
        },
        {
            "id": "seen-user",
            "category": "training_question",
            "question": (
                "Which command prints my current Linux user name?"
            ),
            "expected": "whoami",
        },
        {
            "id": "new-wording",
            "category": "new_wording_of_trained_task",
            "question": (
                "Show the absolute path of my current directory on Linux."
            ),
            "expected": "pwd",
        },
        {
            "id": "hostname",
            "category": "validation_task",
            "question": (
                "Which command displays the Linux host name?"
            ),
            "expected": "hostname",
        },
        {
            "id": "unknown-policy",
            "category": "out_of_scope_probe",
            "question": (
                "What is the incident escalation phone number "
                "at fictional Meridian Labs?"
            ),
            "expected": (
                "Acknowledge that no authoritative number was supplied."
            ),
        },
    ]

    settings = {
        "max_new_tokens": 24,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    records = []

    for case in cases:
        # Match the training prompt format exactly.
        prompt = (
            "Answer the IT support question with the command only.\n"
            f"Question: {case['question']}\n"
            "Answer:\n"
        )

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_length = inputs["input_ids"].shape[1]

        for condition in ("base", "adapter"):
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

            generated_ids = output[0, prompt_length:]

            answer = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            ended_with_eos = (
                generated_ids.numel() > 0
                and generated_ids[-1].item()
                == tokenizer.eos_token_id
            )

            # Exact matching is only a narrow diagnostic.
            # The policy probe requires human review.
            exact_match = (
                answer == case["expected"]
                if case["category"] != "out_of_scope_probe"
                else None
            )

            record = {
                "case_id": case["id"],
                "category": case["category"],
                "condition": condition,
                "prompt": prompt,
                "answer": answer,
                "expected": case["expected"],
                "exact_match": exact_match,
                "ended_with_eos": ended_with_eos,
                "generated_tokens": generated_ids.numel(),
                "human_review": "pending",
            }

            records.append(record)

            print(f"\n{case['id']} | {condition}")
            print("Answer:", repr(answer))
            print("Exact match:", exact_match)
            print("Ended with EOS:", ended_with_eos)

    results = {
        "model_id": model_id,
        "model_revision": revision,
        "saved_training_loss": saved_loss,
        "restored_training_loss": restored_loss,
        "reload_check_passed": loss_matches,
        "generation_settings": settings,
        "evaluation_scope": (
            "Exploratory development evaluation. "
            "Training questions and visible validation material "
            "are not a sealed test set."
        ),
        "records": records,
    }

    output_path = ROOT / "results" / "adapter_evaluation.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()