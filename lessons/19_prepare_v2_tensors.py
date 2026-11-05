import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "command_assistant_v2"


def main():
    revision_config = json.loads(
        (ROOT / "configs" / "model_revision.json").read_text(
            encoding="utf-8"
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        revision_config["model_id"],
        revision=revision_config["revision"],
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    instruction_path = DATA_DIR / "instruction.txt"
    training_path = DATA_DIR / "train.jsonl"

    shared_instruction = instruction_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    records = [
        json.loads(line)
        for line in training_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    prepared = []
    maximum_length = 128

    for record in records:
        prompt = (
            shared_instruction
            + "\n\n"
            + f"Question: {record['instruction']}\n"
        )

        if record["input"]:
            prompt += f"Context: {record['input']}\n"

        prompt += "Answer:\n"

        # Encode segments separately, matching our earlier lesson.
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        answer_ids = tokenizer.encode(
            record["output"],
            add_special_tokens=False,
        ) + [tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        if len(input_ids) > maximum_length:
            raise ValueError(
                f"{record['id']}: {len(input_ids)} tokens "
                f"exceeds limit {maximum_length}"
            )

        prepared.append({
            "id": record["id"],
            "prompt": prompt,
            "input_ids": input_ids,
            "labels": labels,
            "prompt_length": len(prompt_ids),
            "sequence_length": len(input_ids),
        })

    if not prepared:
        raise ValueError("Training dataset is empty")

    batch_length = max(
        record["sequence_length"] for record in prepared
    )

    input_rows = []
    label_rows = []
    mask_rows = []
    supervised_counts = []

    for record in prepared:
        padding_length = batch_length - record["sequence_length"]

        inputs = (
            record["input_ids"]
            + [tokenizer.pad_token_id] * padding_length
        )
        labels = record["labels"] + [-100] * padding_length
        mask = (
            [1] * record["sequence_length"]
            + [0] * padding_length
        )

        # Check prompt masking, EOS supervision, and padding masking.
        assert all(
            value == -100
            for value in labels[:record["prompt_length"]]
        )

        eos_position = record["sequence_length"] - 1
        assert labels[eos_position] == tokenizer.eos_token_id
        assert mask[eos_position] == 1

        assert all(
            label == -100
            for label, active in zip(labels, mask)
            if active == 0
        )

        supervised_count = sum(
            label != -100 for label in labels
        )
        assert supervised_count > 0

        input_rows.append(inputs)
        label_rows.append(labels)
        mask_rows.append(mask)
        supervised_counts.append(supervised_count)

        print(
            f"{record['id']} | "
            f"sequence tokens={record['sequence_length']} | "
            f"supervised tokens={supervised_count}"
        )

    report = {
        "model_id": revision_config["model_id"],
        "model_revision": revision_config["revision"],
        "shared_instruction": shared_instruction,
        "prompt_format": (
            "shared instruction + blank line + Question line + "
            "optional Context line + Answer:\\n"
        ),
        "training_file_sha256": hashlib.sha256(
            training_path.read_bytes()
        ).hexdigest(),
        "instruction_file_sha256": hashlib.sha256(
            instruction_path.read_bytes()
        ).hexdigest(),
        "record_ids": [record["id"] for record in prepared],
        "batch_shape": [len(prepared), batch_length],
        "sequence_lengths": [
            record["sequence_length"] for record in prepared
        ],
        "supervised_tokens_per_example": supervised_counts,
        "input_ids": input_rows,
        "attention_mask": mask_rows,
        "labels": label_rows,
        "checks_passed": True,
        "status": "training tensors prepared; no weights updated",
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "v2_training_tensors.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBatch shape:", report["batch_shape"])
    print("Training examples:", len(prepared))
    print("Total supervised tokens:", sum(supervised_counts))
    print("All label checks passed.")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()