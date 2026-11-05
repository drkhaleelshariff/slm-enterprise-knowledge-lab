import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
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
        raise ValueError("An EOS token is required")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files=str(
            ROOT / "data" / "teaching_splits" / "train.jsonl"
        ),
        split="train",
    )

    maximum_length = 128

    def prepare(record):
        prompt = (
            "Answer the IT support question with the command only.\n"
            f"Question: {record['instruction']}\n"
        )

        if record["input"]:
            prompt += f"Context: {record['input']}\n"

        prompt += "Answer:\n"

        # Encode the two segments separately by design.
        # We will use the same prompt construction at inference time.
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        answer_ids = tokenizer.encode(
            record["output"],
            add_special_tokens=False,
        )

        answer_ids.append(tokenizer.eos_token_id)

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        # Reject overlength records rather than silently removing
        # their answer or EOS supervision.
        if len(input_ids) > maximum_length:
            raise ValueError(
                f"{record['id']} exceeds {maximum_length} tokens"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_length": len(prompt_ids),
        }

    prepared = dataset.map(prepare)

    # Pad this small batch to its longest sequence.
    batch_length = max(
        len(record["input_ids"])
        for record in prepared
    )

    input_rows = []
    label_rows = []
    mask_rows = []

    for record in prepared:
        sequence_length = len(record["input_ids"])
        padding_length = batch_length - sequence_length

        input_rows.append(
            record["input_ids"]
            + [tokenizer.pad_token_id] * padding_length
        )

        label_rows.append(
            record["labels"] + [-100] * padding_length
        )

        mask_rows.append(
            [1] * sequence_length + [0] * padding_length
        )

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long)

    print("Input shape:", list(input_ids.shape))
    print("Labels shape:", list(labels.shape))

    supervised_counts = (labels != -100).sum(dim=1)
    print("Supervised tokens per example:", supervised_counts.tolist())

    first = prepared[0]
    prompt_length = first["prompt_length"]

    print("\nFirst example:")
    print("Question:", first["instruction"])
    print("Expected answer:", first["output"])

    supervised_ids = labels[0][labels[0] != -100].tolist()

    print(
        "Supervised text, including EOS:",
        repr(
            tokenizer.decode(
                supervised_ids,
                skip_special_tokens=False,
            )
        ),
    )

    print("\nToken-by-token view:")
    print(f"{'Position':>8} {'Token piece':>24} {'Loss?':>8}")

    for position in range(len(first["input_ids"])):
        token_id = input_ids[0, position].item()
        piece = tokenizer.convert_ids_to_tokens(token_id)
        included = labels[0, position].item() != -100

        print(
            f"{position:8d} {repr(piece):>24} "
            f"{str(included):>8}"
        )

    for index, record in enumerate(prepared):
        prefix_length = record["prompt_length"]
        real_length = len(record["input_ids"])

        assert torch.all(labels[index, :prefix_length] == -100)
        assert labels[index, real_length - 1].item() == tokenizer.eos_token_id

    assert torch.all(labels[attention_mask == 0] == -100)
    assert torch.all(supervised_counts > 0)

    results = {
        "model_id": revision_config["model_id"],
        "model_revision": revision_config["revision"],
        "status": "answer-only SFT label preparation; no training",
        "batch_shape": list(input_ids.shape),
        "supervised_tokens_per_example": supervised_counts.tolist(),
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
        "checks_passed": True,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "sft_labels.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAll label checks passed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()