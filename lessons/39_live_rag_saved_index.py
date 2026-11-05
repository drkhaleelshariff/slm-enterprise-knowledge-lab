import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from saved_index import load_verified_index

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.50


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    torch.set_num_threads(2)

    chunk_path = (
        ROOT / "data" / "knowledge_chunks" / "linux_commands.jsonl"
    )

    chunks = [
        json.loads(line)
        for line in chunk_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not chunks:
        raise ValueError("No knowledge chunks found")

    for chunk in chunks:
        actual_hash = hashlib.sha256(
            (ROOT / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if actual_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Rerun Lesson 27 first.")

    embedding_config = read_json(
        ROOT / "configs" / "embedding_model_revision.json"
    )

    print("Loading embedding model...", flush=True)

    embedder = SentenceTransformer(
        embedding_config["model_id"],
        revision=embedding_config["revision"],
        device="cpu",
    )

    saved_vectors, chunks, index_manifest = load_verified_index(ROOT)

    if index_manifest["embedding_model"] != embedding_config:
        raise ValueError(
            "Saved index and query encoder configurations differ"
        )

    document_vectors = torch.from_numpy(saved_vectors)

    print("Verified saved index loaded:", list(document_vectors.shape))
    print("Document embeddings recomputed: False")

    training = read_json(
        ROOT / "results" / "v3_lora_training.json"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training["model_id"],
        revision=training["model_revision"],
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading V3 language model...", flush=True)

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

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    session_records = []

    print("\nReady. Type a question, or exit to finish.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # Check source freshness again in case it changed during use.
        for chunk in chunks:
            current_hash = hashlib.sha256(
                (ROOT / chunk["source_path"]).read_bytes()
            ).hexdigest()

            if current_hash != chunk["source_sha256"]:
                raise ValueError(
                    "Source changed during this session. "
                    "Rerun Lesson 27 and restart this script."
                )

        query_vector = embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        scores = (query_vector @ document_vectors.T)[0]
        best_index = scores.argmax().item()
        similarity = scores[best_index].item()
        chunk = chunks[best_index]

        print(
            f"Nearest section: {chunk['section_title']} "
            f"(similarity={similarity:.4f})"
        )

        record = {
            "question": question,
            "top_chunk_id": chunk["chunk_id"],
            "similarity": similarity,
            "threshold": THRESHOLD,
        }

        if similarity < THRESHOLD:
            answer = (
                "No evidence passed the retrieval gate. "
                "Please clarify the question or supply a relevant source."
            )

            print("Application response:", answer)

            record.update({
                "model_called": False,
                "answer": answer,
                "supplied_evidence": None,
            })

        else:
            prompt = (
                training["shared_instruction"]
                + "\nUse supplied context as evidence when present."
                + f"\n\nQuestion: {question}\n"
                + f"Context: {chunk['text']}\n"
                + "Answer:\n"
            )

            inputs = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )

            # Explicit teaching limit; do not silently truncate evidence.
            if inputs["input_ids"].shape[1] > 256:
                print("Question and evidence exceed this lesson's token limit.")
                continue

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

            answer = tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            ).strip()

            evidence = {
                key: chunk[key]
                for key in (
                    "chunk_id",
                    "source_path",
                    "source_sha256",
                    "start_line",
                    "end_line",
                )
            }

            print("Model answer:", repr(answer))
            print(
                "Evidence supplied:",
                f"{chunk['source_path']}:"
                f"{chunk['start_line']}-{chunk['end_line']}",
            )
            print("Answer support has not been automatically verified.")

            record.update({
                "model_called": True,
                "prompt": prompt,
                "answer": answer,
                "ended_with_eos": (
                    new_ids.numel() > 0
                    and new_ids[-1].item() == tokenizer.eos_token_id
                ),
                "supplied_evidence": evidence,
                "human_review": "pending",
            })

        session_records.append(record)

    report = {
        "generation_model": training["model_id"],
        "generation_model_revision": training["model_revision"],
        "adapter": "smollm2-command-v3",
        "embedding_model": embedding_config,
        "chunk_file_sha256": hashlib.sha256(
            chunk_path.read_bytes()
        ).hexdigest(),
        "threshold_status": "provisional; development-selected",
        "document_embeddings_recomputed": False,
        "index_directory": "indexes/linux_commands_v1",
        "index_manifest": index_manifest,
        "records": session_records,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = results_dir / f"saved_index_rag_{timestamp}.json"

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSession saved to:", output_path)


if __name__ == "__main__":
    main()