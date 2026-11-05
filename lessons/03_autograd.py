import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # A single training example: input 2 should produce output 6.
    x = torch.tensor(2.0)
    target = torch.tensor(6.0)

    # This is the parameter we want to learn.
    weight = torch.tensor(1.0, requires_grad=True)

    learning_rate = 0.1
    history = []

    print("Learning the relationship: output = weight * input")
    print("Training example: input=2, target=6")
    print(f"Initial weight: {weight.item():.6f}\n")

    for step in range(1, 11):
        # 1. Forward pass: calculate a prediction.
        prediction = weight * x

        # 2. Calculate the error.
        loss = (prediction - target).pow(2)

        # 3. Backward pass: calculate the gradient.
        loss.backward()

        # Capture values before updating the parameter.
        weight_before = weight.item()
        gradient = weight.grad.item()

        history.append(
            {
                "step": step,
                "weight_before_update": weight_before,
                "prediction": prediction.item(),
                "loss": loss.item(),
                "gradient": gradient,
            }
        )

        print(
            f"Step {step:02d} | "
            f"weight={weight_before:.6f} | "
            f"prediction={prediction.item():.6f} | "
            f"loss={loss.item():.8f} | "
            f"gradient={gradient:.6f}"
        )

        # 4. Update the parameter without tracking the update
        # as part of a new computational graph.
        with torch.no_grad():
            weight -= learning_rate * weight.grad

        # 5. Clear the gradient before the next iteration.
        weight.grad = None

    # Evaluate the final weight after the last update.
    with torch.no_grad():
        final_prediction = weight * x
        final_loss = (final_prediction - target).pow(2)

    print(f"\nFinal weight: {weight.item():.6f}")
    print(f"Final prediction: {final_prediction.item():.6f}")
    print(f"Final loss: {final_loss.item():.10f}")

    results = {
        "input": x.item(),
        "target": target.item(),
        "learning_rate": learning_rate,
        "history": history,
        "final_weight": weight.item(),
        "final_prediction": final_prediction.item(),
        "final_loss": final_loss.item(),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "autograd.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()