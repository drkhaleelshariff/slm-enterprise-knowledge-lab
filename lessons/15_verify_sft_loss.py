import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_num_threads(2)

    prepared = json.loads(
        (ROOT / "results" / "sft_labels.json").read_text(
            encoding="utf-8"
        )
    )

    input_ids = torch.tensor(
        prepared["input_ids"],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        prepared["attention_mask"],
        dtype=torch.long,
    )

    labels = torch.tensor(
        prepared["labels"],
        dtype=torch.long,
    )

    print("Loading the pinned model on CPU...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        prepared["model_id"],
        revision=prepared["model_revision"],
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model.eval()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        # The logit at position t predicts the token at t + 1.
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        manual_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )

        # Extract individual target-token losses.
        token_losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shifted_labels)

        supervised = shifted_labels != -100

        per_example_losses = (
            (token_losses * supervised).sum(dim=1)
            / supervised.sum(dim=1)
        )

    model_loss = outputs.loss.item()
    independently_calculated_loss = manual_loss.item()
    difference = abs(model_loss - independently_calculated_loss)

    matches = torch.allclose(
        outputs.loss,
        manual_loss,
        atol=1e-5,
        rtol=1e-5,
    )

    print("\nModel's internal loss:", model_loss)
    print("Manually shifted loss:", independently_calculated_loss)
    print("Absolute difference:", difference)
    print("Losses match:", matches)

    print(
        "\nSupervised targets per example:",
        supervised.sum(dim=1).tolist(),
    )

    print(
        "Per-example answer loss:",
        per_example_losses.tolist(),
    )

    assert matches, "Internal and manual losses differ"
    assert supervised.sum().item() == 12
    assert torch.isfinite(outputs.loss)

    results = {
        "model_id": prepared["model_id"],
        "model_revision": prepared["model_revision"],
        "status": "pre-adaptation answer-only loss verification",
        "weights_updated": False,
        "internal_loss": model_loss,
        "manual_loss": independently_calculated_loss,
        "absolute_difference": difference,
        "losses_match": matches,
        "supervised_target_count": supervised.sum().item(),
        "per_example_answer_loss": per_example_losses.tolist(),
    }

    output_path = ROOT / "results" / "sft_loss_verification.json"

    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAll loss checks passed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()