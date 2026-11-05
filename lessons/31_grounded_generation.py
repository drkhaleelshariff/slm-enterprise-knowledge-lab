import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)

    gate = json.loads(
        (ROOT / "results" / "retrieval_gate.json").read_text(
            encoding="utf-8"
        )
    )

    training = json.loads(
        (ROOT / "results" / "v3_lora_training.json").read_text(
            encoding="utf-8"
        )
    )

    # Check source freshness before using saved evidence.
    for record in gate["records"]:
        chunk = record["selected_chunk"]

        if chunk is not None:
            current_hash = hashlib.sha256(
                (ROOT / chunk["source_path"]).read_bytes()
            ).hexdigest()

            if current_hash != chunk["source_sha256"]:
                raise ValueError(
                    "Source changed. Rerun Lessons 27, 29, and 30."
                )

    tokenizer = AutoTokenizer.from_pretrained(
        training["model_id"],
        revision=training["model_revision"],
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading V3 adapter on CPU...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        training["model_id"],
        revision=training["model_revision"],
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu")

    model = PeftModel.from_pretrained(
        base,
        ROOT / "adapters" / "smollm2-command-v3",
        is_trainable=False,
    )
    model.eval()

    # Use identical instructions in both generation conditions.
    instruction = (
        training["shared_instruction"]
        + "\nUse supplied context as evidence when present."
    )

    def generate(question, context=None):
        prompt = instruction + f"\n\nQuestion: {question}\n"

        if context is not None:
            prompt += f"Context: {context}\n"

        prompt += "Answer:\n"

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output[0, inputs["input_ids"].shape[1]:]

        return {
            "prompt": prompt,
            "answer": tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            ).strip(),
            "ended_with_eos": (
                new_ids.numel() > 0
                and new_ids[-1].item() == tokenizer.eos_token_id
            ),
        }

    records = []

    for record in gate["records"]:
        question = record["query"]
        chunk = record["selected_chunk"]

        print(f"\nQuestion: {question}")

        if not record["passes_gate"]:
            message = (
                "No evidence passed the retrieval gate. "
                "Please clarify the question or supply a relevant source."
            )

            print("Application response:", message)

            records.append({
                "query": question,
                "route": "application_withheld",
                "model_called": False,
                "response": message,
            })
            continue

        without_context = generate(question)
        with_context = generate(question, chunk["text"])

        # Reference extraction is only for this structured teaching corpus.
        commands = [
            line.removeprefix("Command: ").strip()
            for line in chunk["text"].splitlines()
            if line.startswith("Command: ")
        ]

        if len(commands) != 1:
            raise ValueError("Expected one command in the selected chunk")

        reference_command = commands[0]

        print("Without context:", repr(without_context["answer"]))
        print("With context:", repr(with_context["answer"]))
        print("Reference command:", reference_command)

        context_match = with_context["answer"] == reference_command
        print("Exact reference match:", context_match)

        # This is supplied-evidence provenance, not a model-generated
        # citation or proof that the generated answer is supported.
        evidence_source = {
            key: chunk[key]
            for key in (
                "chunk_id",
                "source_path",
                "source_sha256",
                "start_line",
                "end_line",
            )
        }

        records.append({
            "query": question,
            "route": "generation_comparison",
            "model_called": True,
            "without_context": without_context,
            "with_context": with_context,
            "reference_command": reference_command,
            "without_context_exact_match": (
                without_context["answer"] == reference_command
            ),
            "with_context_exact_match": context_match,
            "supplied_evidence": evidence_source,
            "human_review": "pending",
        })

    report = {
        "model_id": training["model_id"],
        "model_revision": training["model_revision"],
        "retrieval_threshold": gate["threshold"],
        "status": "generation over saved retrieval results",
        "records": records,
        "limitations": [
            "Three answerable development probes only.",
            "Threshold was selected using these observed examples.",
            "V3 was not trained on examples containing supplied context.",
            "Evidence provenance is not proof of answer groundedness.",
            "No live retrieval or automatic citation validation.",
        ],
    }

    output_path = ROOT / "results" / "grounded_generation.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()