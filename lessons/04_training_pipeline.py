import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(42)

    # Synthetic relationship: y = 3x + 2
    # Shape [5, 1]: five examples, one feature per example.
    inputs = torch.tensor([
        [-2.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
    ])

    targets = 3 * inputs + 2

    dataset = TensorDataset(inputs, targets)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    # One input feature, one output.
    # Linear learns both a weight and a bias: prediction = wx + b.
    model = nn.Linear(in_features=1, out_features=1)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    print("Learning: y = 3x + 2")
    print("Training examples:", len(dataset))
    print("Batches per epoch:", len(loader))
    print("Initial weight:", model.weight.item())
    print("Initial bias:", model.bias.item())

    history = []

    for epoch in range(1, 101):
        model.train()

        total_squared_error = 0.0
        examples_seen = 0

        for batch_inputs, batch_targets in loader:
            # Clear gradients from the previous batch.
            optimizer.zero_grad(set_to_none=True)

            # Forward pass.
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)

            # Calculate gradients.
            loss.backward()

            # Update weight and bias.
            optimizer.step()

            batch_size = batch_inputs.shape[0]
            total_squared_error += loss.item() * batch_size
            examples_seen += batch_size

        epoch_loss = total_squared_error / examples_seen

        history.append({
            "epoch": epoch,
            "training_mse": epoch_loss,
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"training MSE={epoch_loss:.8f}"
            )

    # These inputs were not used for training.
    test_inputs = torch.tensor([
        [-1.5],
        [0.5],
        [3.0],
    ])
    expected_outputs = 3 * test_inputs + 2

    model.eval()

    with torch.no_grad():
        test_predictions = model(test_inputs)
        test_mse = loss_function(
            test_predictions, expected_outputs
        ).item()

    print(f"\nLearned weight: {model.weight.item():.6f}")
    print(f"Learned bias: {model.bias.item():.6f}")

    print("\nEvaluation on unseen inputs:")

    for x, expected, predicted in zip(
        test_inputs.flatten().tolist(),
        expected_outputs.flatten().tolist(),
        test_predictions.flatten().tolist(),
    ):
        print(
            f"Input={x:5.1f} | "
            f"Expected={expected:6.2f} | "
            f"Predicted={predicted:9.6f}"
        )

    print(f"\nTest MSE: {test_mse:.10f}")

    results = {
        "seed": 42,
        "relationship": "y = 3x + 2",
        "batch_size": 2,
        "epochs": 100,
        "learning_rate": 0.05,
        "learned_weight": model.weight.item(),
        "learned_bias": model.bias.item(),
        "test_inputs": test_inputs.flatten().tolist(),
        "expected_outputs": expected_outputs.flatten().tolist(),
        "predictions": test_predictions.flatten().tolist(),
        "test_mse": test_mse,
        "history": history,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "training_pipeline.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()