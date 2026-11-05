import os

# Set the cache location before importing Hugging Face libraries.
os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    # Downloads tokenizer assets, not model weights.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Some causal language-model tokenizers lack a padding token.
    # Reusing EOS avoids adding a new vocabulary entry here.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    texts = [
        "Check disk space.",
        "Please check the installed Python version.",
        "Troubleshooting tokenization and reproducibility.",
    ]

    print("Tokenizer:", MODEL_ID)
    print("Tokenizer class:", type(tokenizer).__name__)
    print("Vocabulary entries:", len(tokenizer))
    print("Special tokens:", tokenizer.special_tokens_map)
    print("Padding token ID:", tokenizer.pad_token_id)

    examples = []

    print("\n1. Text, token pieces, and IDs")

    for text in texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        pieces = tokenizer.convert_ids_to_tokens(token_ids)

        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        print(f"\nText: {text}")
        print("Pieces:", pieces)
        print("IDs:", token_ids)
        print("Token count:", len(token_ids))
        print("Decoded:", decoded)

        examples.append({
            "text": text,
            "pieces": pieces,
            "ids": token_ids,
            "decoded": decoded,
        })

    print("\n2. Batch padding")

    batch = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    print("Input IDs:")
    print(batch["input_ids"])

    print("\nAttention mask:")
    print(batch["attention_mask"])

    print("\nBatch shape:", list(batch["input_ids"].shape))
    print(
        "Real tokens per example:",
        batch["attention_mask"].sum(dim=1).tolist(),
    )

    # Prepare labels for a future Hugging Face causal-LM lesson.
    # Keep labels aligned with input_ids: the model will shift
    # predictions and targets internally.
    labels = batch["input_ids"].clone()

    # Mask padding by POSITION, not by token ID.
    # This preserves genuine EOS labels if PAD and EOS share an ID.
    labels[batch["attention_mask"] == 0] = -100

    print("\nLabels with padded positions ignored:")
    print(labels)

    print("\n3. Truncation")

    long_text = (
        "Check disk space and inspect running processes. "
    ) * 10

    full_ids = tokenizer.encode(
        long_text,
        add_special_tokens=False,
    )

    shortened = tokenizer(
        long_text,
        add_special_tokens=False,
        truncation=True,
        max_length=12,
    )

    print("Original token count:", len(full_ids))
    print("Truncated token count:", len(shortened["input_ids"]))
    print(
        "Truncated text:",
        tokenizer.decode(shortened["input_ids"]),
    )

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert len(shortened["input_ids"]) == 12

    for index, example in enumerate(examples):
        actual_length = batch["attention_mask"][index].sum().item()
        assert actual_length == len(example["ids"])

    # Save tokenizer assets locally for reuse without downloading again.
    local_tokenizer_dir = ROOT / "models" / "smollm2-tokenizer"
    tokenizer.save_pretrained(local_tokenizer_dir)

    results = {
        "model_id": MODEL_ID,
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_entries": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "examples": examples,
        "batch_shape": list(batch["input_ids"].shape),
        "input_ids": batch["input_ids"].tolist(),
        "attention_mask": batch["attention_mask"].tolist(),
        "labels": labels.tolist(),
        "original_long_text_token_count": len(full_ids),
        "truncated_token_count": len(shortened["input_ids"]),
        "checks_passed": True,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "real_tokenizer.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nAll tokenizer checks passed.")
    print(f"Tokenizer saved to: {local_tokenizer_dir}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()