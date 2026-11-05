import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class TinyLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, context_length: int):
        super().__init__()

        self.context_length = context_length
        embedding_dimension = 32

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dimension,
        )

        # A self-attention block with a causal mask acts as
        # a decoder-style block without encoder cross-attention.
        block = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )

        self.transformer_block = block

        self.output_layer = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]

        if sequence_length > self.context_length:
            raise ValueError("Input exceeds the context length")

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        hidden = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # Boolean True means this attention position is blocked
        # for TransformerEncoderLayer.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )

        padding_mask = token_ids == 0

        hidden = self.transformer_block(
            hidden,
            src_mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        return self.output_layer(hidden)


def main() -> None:
    torch.manual_seed(42)

    # Keep CPU use modest on the laptop.
    torch.set_num_threads(2)

    words = [
        "<pad>", "<bos>", "<eos>",
        "check", "disk", "space",
        "python", "version",
        "list", "running", "processes",
    ]

    vocabulary = {
        word: index for index, word in enumerate(words)
    }

    sequences = [
        ["<bos>", "check", "disk", "space", "<eos>"],
        ["<bos>", "check", "python", "version", "<eos>"],
        ["<bos>", "list", "running", "processes", "<eos>"],
    ]

    encoded = torch.tensor([
        [vocabulary[word] for word in sequence]
        for sequence in sequences
    ], dtype=torch.long)

    inputs = encoded[:, :-1]
    targets = encoded[:, 1:]

    model = TinyLanguageModel(
        vocabulary_size=len(vocabulary),
        context_length=8,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0,
    )

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Device: CPU")
    print("Training sequences:", len(sequences))
    print("Input shape:", list(inputs.shape))
    print("Trainable parameters:", parameter_count)

    # Evaluation measures the same fixed training examples
    # with parameters held constant.
    def evaluate_training_loss():
        model.eval()

        with torch.no_grad():
            logits = model(inputs)
            loss = loss_function(
                logits.reshape(-1, len(vocabulary)),
                targets.reshape(-1),
            )

        return loss.item()

    initial_loss = evaluate_training_loss()
    print(f"Initial loss: {initial_loss:.6f}")

    history = []

    for step in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = loss_function(
            logits.reshape(-1, len(vocabulary)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            measured_loss = evaluate_training_loss()

            history.append({
                "step": step,
                "training_loss_after_update": measured_loss,
            })

            print(
                f"Step {step:03d} | "
                f"loss={measured_loss:.6f}"
            )

    final_loss = evaluate_training_loss()

    # Check that changing a future token cannot change
    # logits at an earlier position.
    model.eval()

    with torch.no_grad():
        original = inputs[:1].clone()
        modified = original.clone()
        modified[0, 2] = vocabulary["python"]

        original_logits = model(original)
        modified_logits = model(modified)

        causal_check = torch.allclose(
            original_logits[:, :2],
            modified_logits[:, :2],
            atol=1e-5,
            rtol=1e-5,
        )

    print("\nFuture-token isolation check:", causal_check)

    def generate(prompt_words):
        ids = [
            vocabulary["<bos>"],
            *[vocabulary[word] for word in prompt_words],
        ]

        model.eval()

        with torch.no_grad():
            for _ in range(5):
                if len(ids) >= model.context_length:
                    break

                batch = torch.tensor([ids], dtype=torch.long)
                next_logits = model(batch)[0, -1]
                next_id = next_logits.argmax().item()

                ids.append(next_id)

                if next_id == vocabulary["<eos>"]:
                    break

        return [words[token_id] for token_id in ids]

    generations = {}

    for prompt in [
        ["check", "disk"],
        ["check", "python"],
        ["list"],
    ]:
        generated_words = generate(prompt)
        prompt_text = " ".join(prompt)

        generations[prompt_text] = " ".join(generated_words)

        print(f"\nPrompt: {prompt_text}")
        print("Generated:", generations[prompt_text])

    assert causal_check, "Future tokens affected earlier predictions"
    assert final_loss < initial_loss, "Training did not reduce loss"
    # Save learned weights and the configuration needed to rebuild
    # this model. This file is a local artifact, excluded from Git.
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "tiny_transformer.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "context_length": model.context_length,
            "seed": 42,
        },
        checkpoint_path,
    )

    # Reconstruct the architecture and restore learned parameters.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    restored_model = TinyLanguageModel(
        vocabulary_size=len(checkpoint["vocabulary"]),
        context_length=checkpoint["context_length"],
    )

    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    restored_model.eval()

    with torch.no_grad():
        original_logits = model(inputs)
        restored_logits = restored_model(inputs)

        maximum_difference = (
            original_logits - restored_logits
        ).abs().max().item()

        reload_matches = torch.allclose(
            original_logits,
            restored_logits,
            atol=1e-6,
            rtol=1e-6,
        )

    assert reload_matches, "Reloaded model predictions changed"

    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print("Reloaded predictions match:", reload_matches)
    print("Maximum logit difference:", maximum_difference)

    results = {
        "status": "toy transformer trained on three sequences",
        "evaluation_scope": "training examples and seen prefixes",
        "seed": 42,
        "parameter_count": parameter_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "future_token_isolation_passed": causal_check,
        "history": history,
        "generations": generations,
        "checkpoint_reload_passed": reload_matches,
        "checkpoint_max_logit_difference": maximum_difference,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "tiny_transformer.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()