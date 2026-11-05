import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

import torch
import sentence_transformers
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


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
        raise ValueError("No chunks found")

    for chunk in chunks:
        actual_hash = hashlib.sha256(
            (ROOT / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if actual_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Rerun Lesson 27.")

    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    revision_path = config_dir / "embedding_model_revision.json"

    if revision_path.exists():
        configuration = json.loads(
            revision_path.read_text(encoding="utf-8")
        )
        if configuration["model_id"] != MODEL_ID:
            raise ValueError("Embedding model configuration mismatch")
        revision = configuration["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha
        if not revision:
            raise ValueError("Could not resolve model revision")

        revision_path.write_text(
            json.dumps(
                {"model_id": MODEL_ID, "revision": revision},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    print("Loading embedding model on CPU...", flush=True)

    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        device="cpu",
    )

    document_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    print("Document embedding shape:", list(document_vectors.shape))

    queries = [
        "Which Linux command prints the kernel release?",
        "Show filesystem space usage.",
        "Where am I?",
        "In a Linux terminal, how do I find my current folder's path?",
        "Meridian escalation telephone?",
    ]

    records = []

    for query in queries:
        query_vector = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # For unit-length vectors, dot product equals cosine similarity.
        scores = (query_vector @ document_vectors.T)[0]
        top_scores, top_indices = torch.topk(
            scores,
            k=min(2, len(chunks)),
        )

        hits = []

        print(f"\nQuestion: {query}")

        for score, index in zip(
            top_scores.tolist(),
            top_indices.tolist(),
        ):
            chunk = chunks[index]

            hit = {
                **chunk,
                "cosine_similarity": score,
            }
            hits.append(hit)

            print(
                f"  {chunk['section_title']} | "
                f"similarity={score:.4f}"
            )
            print(
                f"  Source: {chunk['source_path']}:"
                f"{chunk['start_line']}-{chunk['end_line']}"
            )

        records.append({
            "query": query,
            "hits": hits,
        })

    report = {
        "embedding_model": MODEL_ID,
        "embedding_model_revision": revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "chunk_file_sha256": hashlib.sha256(
            chunk_path.read_bytes()
        ).hexdigest(),
        "document_embedding_shape": list(document_vectors.shape),
        "method": "cosine similarity using normalized embeddings",
        "top_k": 2,
        "threshold": None,
        "records": records,
        "limitations": [
            "Always returns nearest candidates, even for missing answers.",
            "Similarity is not a probability of correctness.",
            "The question 'Where am I?' is ambiguous.",
            "No answer generation or evidence-sufficiency check performed.",
        ],
    }

    output_path = ROOT / "results" / "semantic_retrieval.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()