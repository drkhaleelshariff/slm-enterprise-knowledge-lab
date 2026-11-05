import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(42)

    # A teaching vocabulary, not a pretrained model's tokenizer.
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "check": 2,
        "disk": 3,
        "space": 4,
        "python": 5,
        "version": 6,
        "please": 7,
    }

    sentences = [
        "check disk space",
        "please check python version",
    ]

    max_length = 5
    pad_id = vocabulary["<pad>"]
    unknown_id = vocabulary["<unk>"]

    def encode(text: str) -> list[int]:
        words = text.lower().split()
        ids = [vocabulary.get(word, unknown_id) for word in words]

        # Truncate, then pad to a fixed length.
        ids = ids[:max_length]
        return ids + [pad_id] * (max_length - len(ids))

    token_ids = torch.tensor(
        [encode(sentence) for sentence in sentences],
        dtype=torch.long,
    )

    attention_mask = (token_ids != pad_id).long()

    embedding_layer = nn.Embedding(
        num_embeddings=len(vocabulary),
        embedding_dim=4,
        padding_idx=pad_id,
    )

    # No training in this lesson.
    with torch.no_grad():
        embeddings = embedding_layer(token_ids)

        # Expand [batch, length] to [batch, length, 1].
        # Broadcasting applies each mask value to all four dimensions.
        expanded_mask = attention_mask.unsqueeze(-1).to(
            embeddings.dtype
        )

        # Average real-token vectors, excluding padding.
        summed_vectors = (embeddings * expanded_mask).sum(dim=1)
        token_counts = expanded_mask.sum(dim=1).clamp_min(1)
        sentence_vectors = summed_vectors / token_counts

    print("Sentences:")
    for sentence in sentences:
        print(" ", sentence)

    print("\nToken IDs:")
    print(token_ids)

    print("\nAttention mask:")
    print(attention_mask)

    print("\nEmbedding table shape:")
    print(list(embedding_layer.weight.shape))

    print("\nToken embedding output shape:")
    print(list(embeddings.shape))

    print("\nVector for 'check':")
    print(embedding_layer.weight[vocabulary["check"]].detach())

    print("\n'check' vector in sentence 1:")
    print(embeddings[0, 0])

    print("\n'check' vector in sentence 2:")
    print(embeddings[1, 1])

    print("\nSame lookup vector?",
          torch.equal(embeddings[0, 0], embeddings[1, 1]))

    print("\nPadding vector:")
    print(embedding_layer.weight[pad_id].detach())

    print("\nSentence vectors using masked mean pooling:")
    print(sentence_vectors)

    print("\nSentence vector shape:")
    print(list(sentence_vectors.shape))

    results = {
        "seed": 42,
        "embedding_status": "random initialization; not trained",
        "vocabulary": vocabulary,
        "sentences": sentences,
        "token_ids": token_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "embedding_table_shape": list(embedding_layer.weight.shape),
        "token_embedding_shape": list(embeddings.shape),
        "sentence_vector_shape": list(sentence_vectors.shape),
        "sentence_vectors": sentence_vectors.tolist(),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "embeddings.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()