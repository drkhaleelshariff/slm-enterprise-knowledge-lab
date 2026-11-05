import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    vocabulary = {
        "<pad>": 0,
        "check": 1,
        "disk": 2,
        "space": 3,
        "<eos>": 4,
    }

    id_to_token = {
        token_id: token
        for token, token_id in vocabulary.items()
    }

    # One sequence, padded to length 6.
    sequence = torch.tensor([
        [1, 2, 3, 4, 0, 0]
    ], dtype=torch.long)

    # Shift by one position for next-token prediction.
    inputs = sequence[:, :-1]
    targets = sequence[:, 1:].clone()

    # Ignore padded targets when calculating loss.
    targets[targets == vocabulary["<pad>"]] = -100

    print("Full sequence:")
    print(sequence)

    print("\nModel input IDs:")
    print(inputs)

    print("\nNext-token targets (-100 means ignored):")
    print(targets)

    print("\nTraining pairs:")
    for input_id, target_id in zip(
        inputs[0].tolist(),
        targets[0].tolist(),
    ):
        target_name = (
            "IGNORED"
            if target_id == -100
            else id_to_token[target_id]
        )
        print(f"{id_to_token[input_id]:>6} -> {target_name}")

    # Shape: [batch=1, positions=5, vocabulary=5].
    # Each position has one raw score per vocabulary token.
    # These are illustrative logits, not outputs from a trained model.
    logits = torch.tensor([
        [
            [0.0, 0.0, 3.0, 0.0, 0.0],  # Predict disk
            [0.0, 0.0, 0.0, 3.0, 0.0],  # Predict space
            [0.0, 0.0, 0.0, 0.0, 3.0],  # Predict eos
            [3.0, 0.0, 0.0, 0.0, 0.0],  # Ignored target
            [3.0, 0.0, 0.0, 0.0, 0.0],  # Ignored target
        ]
    ], requires_grad=True)

    vocabulary_size = len(vocabulary)

    # CrossEntropyLoss expects raw logits, not softmax probabilities.
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    loss = loss_function(
        logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
    )

    # Compare against equal scores for every vocabulary token.
    uniform_logits = torch.zeros_like(logits)

    uniform_loss = loss_function(
        uniform_logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
    )

    probabilities = torch.softmax(logits.detach(), dim=-1)
    predicted_ids = logits.detach().argmax(dim=-1)

    print("\nLogits shape:", list(logits.shape))
    print("Predicted IDs:", predicted_ids.tolist())

    print("\nProbability of the correct token:")
    for position in range(targets.shape[1]):
        target_id = targets[0, position].item()

        if target_id != -100:
            probability = probabilities[
                0, position, target_id
            ].item()

            print(
                f"Position {position}: "
                f"{id_to_token[target_id]} = {probability:.4f}"
            )

    print(f"\nIllustrative prediction loss: {loss.item():.6f}")
    print(f"Uniform prediction loss: {uniform_loss.item():.6f}")

    loss.backward()

    valid_positions = targets != -100
    ignored_gradients = logits.grad[~valid_positions]

    assert torch.all(ignored_gradients == 0)
    assert loss.item() < uniform_loss.item()

    print("\nIgnored positions have zero logit gradients: True")
    print("Illustrative predictions beat uniform scores: True")

    results = {
        "status": "manual logits; no model training",
        "vocabulary": vocabulary,
        "input_ids": inputs.tolist(),
        "targets": targets.tolist(),
        "logits_shape": list(logits.shape),
        "valid_target_count": valid_positions.sum().item(),
        "loss": loss.item(),
        "uniform_loss": uniform_loss.item(),
        "ignored_positions_have_zero_logit_gradients": True,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "next_token_loss.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()