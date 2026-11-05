import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)

    index_dir = ROOT / "indexes" / "linux_commands_corrupt_demo"

    manifest = json.loads(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )

    # Verify the saved vector and metadata files.
    for filename, hash_field in [
        ("vectors.npy", "vectors_sha256"),
        ("chunks.json", "metadata_sha256"),
    ]:
        actual_hash = hashlib.sha256(
            (index_dir / filename).read_bytes()
        ).hexdigest()

        if actual_hash != manifest[hash_field]:
            raise ValueError(f"{filename} differs from the manifest")

    # Check that the original chunk file has not changed.
    chunk_path = ROOT / manifest["chunk_file"]

    if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != (
        manifest["chunk_file_sha256"]
    ):
        raise ValueError("Chunk file changed. Build a new index.")

    vectors = np.load(
        index_dir / "vectors.npy",
        allow_pickle=False,
    )

    chunks = json.loads(
        (index_dir / "chunks.json").read_text(encoding="utf-8")
    )

    if list(vectors.shape) != manifest["vector_shape"]:
        raise ValueError("Unexpected vector shape")

    if str(vectors.dtype) != manifest["vector_dtype"]:
        raise ValueError("Unexpected vector data type")

    if len(chunks) != vectors.shape[0]:
        raise ValueError("Vector and metadata counts differ")

    if [chunk["chunk_id"] for chunk in chunks] != (
        manifest["row_chunk_ids"]
    ):
        raise ValueError("Vector-to-chunk ordering mismatch")

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite vector values")

    if not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError("Document vectors are not normalized")

    for chunk in chunks:
        source_hash = hashlib.sha256(
            (ROOT / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if source_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Build fresh chunks and index.")

    print("Saved index checks passed.")
    print("Loaded document vectors:", list(vectors.shape))
    print("Document embeddings recomputed: False")

    configuration = manifest["embedding_model"]

    model = SentenceTransformer(
        configuration["model_id"],
        revision=configuration["revision"],
        device="cpu",
    )

    query = input("\nEnter your question: ").strip()

    if not query:
        raise ValueError("Question is empty")

    # Only the question is embedded during search.
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if query_vector.shape != (1, vectors.shape[1]):
        raise ValueError("Question embedding dimension mismatch")

    scores = (query_vector @ vectors.T)[0]

    # Stable ordering makes ties deterministic.
    ranked_indices = np.argsort(-scores, kind="stable")[:2]

    hits = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[int(index)]
        score = float(scores[index])

        print(
            f"\n{rank}. {chunk['section_title']} "
            f"| similarity={score:.4f}"
        )
        print(chunk["text"])
        print(
            f"Source: {chunk['source_path']}:"
            f"{chunk['start_line']}-{chunk['end_line']}"
        )

        hits.append({
            "chunk_id": chunk["chunk_id"],
            "section_title": chunk["section_title"],
            "cosine_similarity": score,
        })

    report = {
        "query": query,
        "index_directory": index_dir.relative_to(ROOT).as_posix(),
        "embedding_model": configuration,
        "document_embeddings_recomputed": False,
        "hits": hits,
        "limitation": (
            "Nearest candidates only; no relevance gate or generation."
        ),
    }

    output_path = ROOT / "results" / "saved_index_search.json"

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()
