import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    tokens = ["check", "disk", "space", "<pad>"]

    # Illustrative vectors, manually chosen for this exercise.
    # Shapes: [sequence_length, feature_count] = [4, 2]
    vectors = torch.tensor([
        [1.0, 0.0],   # check
        [0.0, 1.0],   # disk
        [1.0, 1.0],   # space
        [0.0, 0.0],   # padding
    ])

    # Identity projections simplify the arithmetic.
    # Real transformers learn separate Q, K, and V projections.
    identity = torch.eye(2)

    queries = vectors @ identity
    keys = vectors @ identity
    values = vectors @ identity

    key_dimension = keys.shape[-1]

    scores = (queries @ keys.T) / math.sqrt(key_dimension)

    print("Token order:", tokens)
    print("\nInput vectors:")
    print(vectors)

    print("\nScaled attention scores:")
    print(scores)

    # True means a position contains a real token.
    valid_tokens = torch.tensor([True, True, True, False])

    # Mask padding in the KEY positions, which are columns.
    padding_key_mask = ~valid_tokens.unsqueeze(0)

    padding_masked_scores = scores.masked_fill(
        padding_key_mask,
        float("-inf"),
    )

    full_weights = torch.softmax(padding_masked_scores, dim=-1)
    full_outputs = full_weights @ values

    # Decoder-style causal mask:
    # block columns representing positions after the current row.
    sequence_length = len(tokens)

    future_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    combined_mask = padding_key_mask | future_mask

    causal_scores = scores.masked_fill(
        combined_mask,
        float("-inf"),
    )

    causal_weights = torch.softmax(causal_scores, dim=-1)
    causal_outputs = causal_weights @ values

    print("\nAttention weights: padding masked, future allowed")
    print(full_weights)

    print("\nAttention weights: padding and future masked")
    print(causal_weights)

    print("\nCausal attention row sums:")
    print(causal_weights.sum(dim=-1))

    print("\nOutput for 'check' with future tokens allowed:")
    print(full_outputs[0])

    print("\nOutput for 'check' with causal masking:")
    print(causal_outputs[0])

    # Key masking does not remove padded QUERY rows.
    # Explicitly zero that output row for this demonstration.
    causal_outputs = causal_outputs.masked_fill(
        ~valid_tokens.unsqueeze(-1),
        0.0,
    )

    print("\nCausal outputs, with padded query output zeroed:")
    print(causal_outputs)

    # Check the two central masking requirements.
    assert torch.all(causal_weights[:, 3] == 0)
    assert torch.all(causal_weights[future_mask] == 0)

    assert torch.allclose(
        causal_weights.sum(dim=-1),
        torch.ones(sequence_length),
    )

    results = {
        "tokens": tokens,
        "status": "manually chosen vectors; no training",
        "scaled_scores": scores.tolist(),
        "full_attention_weights": full_weights.tolist(),
        "causal_attention_weights": causal_weights.tolist(),
        "causal_outputs": causal_outputs.tolist(),
        "checks": {
            "padding_keys_blocked": True,
            "future_positions_blocked": True,
            "attention_rows_sum_to_one": True,
        },
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "attention.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAll masking checks passed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()