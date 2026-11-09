import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)

    index_dir = ROOT / "indexes" / "linux_commands_faiss_v1"

    manifest = json.loads(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )

    # Check file hashes before deserializing the local index.
    for filename, hash_field in [
        ("index.faiss", "index_sha256"),
        ("chunks.json", "metadata_sha256"),
    ]:
        actual_hash = hashlib.sha256(
            (index_dir / filename).read_bytes()
        ).hexdigest()

        if actual_hash != manifest[hash_field]:
            raise ValueError(f"{filename} differs from the manifest")

    chunk_path = ROOT / manifest["chunk_file"]

    if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != (
        manifest["chunk_file_sha256"]
    ):
        raise ValueError("Chunk file changed. Rebuild the index.")

    chunks = json.loads(
        (index_dir / "chunks.json").read_text(encoding="utf-8")
    )

    if [chunk["chunk_id"] for chunk in chunks] != (
        manifest["row_chunk_ids"]
    ):
        raise ValueError("Chunk ordering differs from the manifest")

    for chunk in chunks:
        actual_hash = hashlib.sha256(
            (ROOT / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if actual_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Rebuild chunks and index.")

    index = faiss.read_index(str(index_dir / "index.faiss"))

    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("Expected IndexFlatIP")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Unexpected search metric")

    if index.d != manifest["dimension"]:
        raise ValueError("Index dimension mismatch")

    if index.ntotal != manifest["vector_count"]:
        raise ValueError("Index count mismatch")

    if index.ntotal != len(chunks) or not chunks:
        raise ValueError("Index and metadata counts are invalid")

    print("Saved FAISS index checks passed.")
    print("Stored vectors:", index.ntotal)
    print("Document embeddings recomputed: False")

    configuration = manifest["embedding_model"]

    model = SentenceTransformer(
        configuration["model_id"],
        revision=configuration["revision"],
        device="cpu",
    )

    question = input("\nEnter your question: ").strip()

    if not question:
        raise ValueError("Question is empty")

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.shape != (1, index.d):
        raise ValueError("Question embedding dimension mismatch")

    if not np.isfinite(query_vector).all():
        raise ValueError("Invalid question embedding")

    scores, ids = index.search(
        query_vector,
        min(2, index.ntotal),
    )

    hits = []

    for rank, (score, row_id) in enumerate(
        zip(scores[0], ids[0]),
        start=1,
    ):
        chunk = chunks[int(row_id)]

        print(
            f"\n{rank}. {chunk['section_title']} "
            f"| similarity={float(score):.4f}"
        )
        print(chunk["text"])
        print(
            f"Source: {chunk['source_path']}:"
            f"{chunk['start_line']}-{chunk['end_line']}"
        )

        hits.append({
            "chunk_id": chunk["chunk_id"],
            "section_title": chunk["section_title"],
            "cosine_similarity": float(score),
        })

    report = {
        "question": question,
        "index_directory": index_dir.relative_to(ROOT).as_posix(),
        "index_sha256": manifest["index_sha256"],
        "embedding_model": configuration,
        "document_embeddings_recomputed": False,
        "hits": hits,
        "limitation": "Ranking only; no relevance gate or generation.",
    }

    output_path = ROOT / "results" / "saved_faiss_search.json"

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()