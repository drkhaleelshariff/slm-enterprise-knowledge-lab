import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)

    prepared = json.loads(
        (ROOT / "results" / "sft_labels.json").read_text(
            encoding="utf-8"
        )
    )

    batch = {
        name: torch.tensor(prepared[name], dtype=torch.long)
        for name in ("input_ids", "attention_mask", "labels")
    }

    print("Loading pinned base model on CPU...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        prepared["model_id"],
        revision=prepared["model_revision"],
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    base_model.config.use_cache = False

    configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(base_model, configuration)
    model.print_trainable_parameters()

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    # Check that only LoRA parameters are trainable.
    assert trainable
    assert all("lora_" in name for name, _ in trainable)

    trainable_count = sum(
        parameter.numel() for _, parameter in trainable
    )
    total_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    # Keep a small frozen-weight sample for an additional check.
    frozen_name, frozen_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.ndim == 2
    )
    frozen_sample_before = (
        frozen_parameter.detach().flatten()[:128].clone()
    )

    def measure_loss():
        model.eval()
        with torch.no_grad():
            return model(**batch, use_cache=False).loss.item()

    initial_loss = measure_loss()
    print(f"\nInitial answer loss: {initial_loss:.6f}")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=0.001,
        weight_decay=0.0,
    )

    history = []
    start = time.perf_counter()

    for step in range(1, 21):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = model(**batch, use_cache=False).loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()

        # Frozen base parameters should not receive gradients.
        if step == 1:
            assert all(
                parameter.grad is None
                for parameter in model.parameters()
                if not parameter.requires_grad
            )

        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            max_norm=1.0,
        )

        optimizer.step()

        if step == 1 or step % 5 == 0:
            measured_loss = measure_loss()

            history.append({
                "step": step,
                "answer_loss_after_update": measured_loss,
            })

            print(
                f"Step {step:02d} | "
                f"answer loss={measured_loss:.6f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    final_loss = measure_loss()

    frozen_sample_unchanged = torch.equal(
        frozen_sample_before,
        frozen_parameter.detach().flatten()[:128],
    )
    assert frozen_sample_unchanged

    adapter_path = ROOT / "adapters" / "smollm2-command-demo"
    adapter_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        adapter_path,
        safe_serialization=True,
    )

    results = {
        "model_id": prepared["model_id"],
        "model_revision": prepared["model_revision"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "rank": 4,
        "alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "learning_rate": 0.001,
        "steps": 20,
        "batch_size": batch["input_ids"].shape[0],
        "trainable_parameters": trainable_count,
        "total_parameters_including_adapters": total_count,
        "trainable_percentage": 100 * trainable_count / total_count,
        "initial_answer_loss": initial_loss,
        "final_answer_loss": final_loss,
        "training_seconds": elapsed,
        "frozen_parameter_sample_name": frozen_name,
        "frozen_parameter_sample_unchanged": frozen_sample_unchanged,
        "history": history,
        "evaluation_scope": (
            "Training loss on four visible examples only. "
            "No validation or generation evaluation performed."
        ),
    }

    output_path = ROOT / "results" / "lora_training.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nInitial answer loss: {initial_loss:.6f}")
    print(f"Final answer loss: {final_loss:.6f}")
    print(f"Training seconds: {elapsed:.2f}")
    print("Frozen weight sample unchanged:", frozen_sample_unchanged)
    print(f"Adapter saved to: {adapter_path}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()