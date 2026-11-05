import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Keep this lesson explicitly on the CPU.
    device = torch.device("cpu")

    print("1. Scalar, vector, and matrix")

    scalar = torch.tensor(5.0, device=device)
    vector = torch.tensor([10.0, 20.0, 30.0], device=device)

    matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        device=device,
    )

    for name, tensor in [
        ("scalar", scalar),
        ("vector", vector),
        ("matrix", matrix),
    ]:
        print(
            f"{name}: shape={list(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}"
        )

    print("\n2. Indexing")
    print("First row:", matrix[0])
    print("Second column:", matrix[:, 1])
    print("Row 2, column 3:", matrix[1, 2].item())

    print("\n3. Element-wise operations")
    print("matrix + 10:\n", matrix + 10)
    print("matrix * 2:\n", matrix * 2)
    print("matrix * matrix:\n", matrix * matrix)

    print("\n4. Broadcasting")

    bias = torch.tensor([10.0, 20.0, 30.0], device=device)
    shifted = matrix + bias

    print("Matrix shape:", list(matrix.shape))
    print("Bias shape:", list(bias.shape))
    print("Result:\n", shifted)

    print("\n5. Matrix multiplication")

    weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        device=device,
    )

    projected = matrix @ weights

    print("Input shape:", list(matrix.shape))
    print("Weights shape:", list(weights.shape))
    print("Output shape:", list(projected.shape))
    print("Output:\n", projected)

    print("\n6. Reductions")
    print("Sum of all values:", matrix.sum().item())
    print("Mean of each column:", matrix.mean(dim=0))
    print("Mean of each row:", matrix.mean(dim=1))

    print("\n7. A batch of token IDs")

    # These are illustrative IDs, not output from a real tokenizer.
    # For this example, we designate 0 as the padding ID.
    token_ids = torch.tensor(
        [
            [11, 25, 36, 0],
            [42, 17, 0, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    attention_mask = (token_ids != 0).long()

    print("Token IDs:\n", token_ids)
    print("Shape:", list(token_ids.shape))
    print("Data type:", token_ids.dtype)
    print("Attention mask:\n", attention_mask)
    print("Non-padding tokens per sequence:", attention_mask.sum(dim=1))

    results = {
        "device": str(device),
        "matrix_shape": list(matrix.shape),
        "broadcast_result": shifted.tolist(),
        "projection_shape": list(projected.shape),
        "projection_result": projected.tolist(),
        "token_batch_shape": list(token_ids.shape),
        "attention_mask": attention_mask.tolist(),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "tensor_basics.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()