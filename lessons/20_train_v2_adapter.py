import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    prepared = json.loads(
        (ROOT / "results" / "v2_training_tensors.json").read_text(
            encoding="utf-8"
        )
    )

    # Ensure the tensors still correspond to the current source files.
    data_dir = ROOT / "data" / "command_assistant_v2"

    for filename, hash_field in [
        ("train.jsonl", "training_file_sha256"),
        ("instruction.txt", "instruction_file_sha256"),
    ]:
        actual_hash = hashlib.sha256(
            (data_dir / filename).read_bytes()
        ).hexdigest()

        if actual_hash != prepared[hash_field]:
            raise ValueError(
                f"{filename} changed. Rerun Lesson 19 first."
            )

    adapter_path = ROOT / "adapters" / "smollm2-command-v2"

    if adapter_path.exists():
        raise FileExistsError(
            f"{adapter_path} already exists. "
            "Choose a new output folder to preserve the previous run."
        )

    print("Loading fresh pinned base model on CPU...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        prepared["model_id"],
        revision=prepared["model_revision"],
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    base_model.config.use_cache = False

    configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        revision=prepared["model_revision"],
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(base_model, configuration)
    model.print_trainable_parameters()

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert all("lora_" in name for name, _ in named_trainable)

    trainable = [
        parameter for _, parameter in named_trainable
    ]

    # Remove padding from each individual example.
    examples = []

    for index, length in enumerate(prepared["sequence_lengths"]):
        example = {
            field: torch.tensor(
                [prepared[field][index][:length]],
                dtype=torch.long,
            )
            for field in ("input_ids", "attention_mask", "labels")
        }

        # Match the causal model's internally shifted targets.
        target_count = (
            example["labels"][:, 1:] != -100
        ).sum().item()

        assert target_count > 0
        examples.append((example, target_count))

    total_targets = sum(count for _, count in examples)

    def measure_loss():
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for example, count in examples:
                loss = model(
                    **example,
                    use_cache=False,
                ).loss.item()

                total_loss += loss * count

        return total_loss / total_targets

    initial_loss = measure_loss()
    print(f"\nInitial training loss: {initial_loss:.6f}", flush=True)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=0.001,
        weight_decay=0.0,
    )

    updates = 20
    history = []
    start = time.perf_counter()

    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for example, count in examples:
            loss = model(**example, use_cache=False).loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")

            # Weight each example by its supervised-token count.
            # This matches the full-set token-average objective.
            scaled_loss = loss * (count / total_targets)
            scaled_loss.backward()

        if update == 1:
            assert all(
                parameter.grad is None
                for parameter in model.parameters()
                if not parameter.requires_grad
            )

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        if update == 1 or update % 5 == 0:
            measured_loss = measure_loss()

            history.append({
                "update": update,
                "training_answer_loss": measured_loss,
            })

            print(
                f"Update {update:02d}/{updates} | "
                f"training loss={measured_loss:.6f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - start
    final_loss = measure_loss()

    model.save_pretrained(
        adapter_path,
        safe_serialization=True,
    )

    report = {
        "model_id": prepared["model_id"],
        "model_revision": prepared["model_revision"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "seed": 42,
        "rank": 4,
        "alpha": 8,
        "learning_rate": 0.001,
        "updates": updates,
        "examples_per_update": len(examples),
        "microbatch_size": 1,
        "supervised_tokens_per_update": total_targets,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "training_seconds": training_seconds,
        "training_file_sha256": prepared["training_file_sha256"],
        "instruction_file_sha256": prepared["instruction_file_sha256"],
        "shared_instruction": prepared["shared_instruction"],
        "history": history,
        "evaluation_scope": (
            "Training loss only. Development examples were not used."
        ),
    }

    report_path = ROOT / "results" / "v2_lora_training.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nFinal training loss: {final_loss:.6f}")
    print(f"Training seconds: {training_seconds:.2f}")
    print("Adapter saved to:", adapter_path)
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()