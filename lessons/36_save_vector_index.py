import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def main():
    chunk_path = (
        ROOT / "data" / "knowledge_chunks" / "linux_commands.jsonl"
    )
    chunk_bytes = chunk_path.read_bytes()

    chunks = [
        json.loads(line)
        for line in chunk_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    if not chunks:
        raise ValueError("No chunks found")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs")

    for chunk in chunks:
        source_hash = hashlib.sha256(
            (ROOT / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if source_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Rerun Lesson 27.")

    configuration = json.loads(
        (ROOT / "configs" / "embedding_model_revision.json").read_text(
            encoding="utf-8"
        )
    )

    print("Loading embedding model...", flush=True)

    model = SentenceTransformer(
        configuration["model_id"],
        revision=configuration["revision"],
        device="cpu",
    )

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    assert vectors.ndim == 2
    assert vectors.shape[0] == len(chunks)
    assert np.isfinite(vectors).all()

    assert np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-5,
    )

    index_dir = ROOT / "indexes" / "linux_commands_v1"

    if index_dir.exists():
        raise FileExistsError(
            "Index folder already exists. "
            "Choose a new folder name for another build."
        )

    index_dir.mkdir(parents=True)

    vector_path = index_dir / "vectors.npy"

    np.save(vector_path, vectors, allow_pickle=False)

    # Store metadata in exactly the same order as the vector rows.
    metadata_path = index_dir / "chunks.json"
    metadata_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format": "normalized float32 NumPy vectors with ordered metadata",
        "embedding_model": configuration,
        "chunk_file": chunk_path.relative_to(ROOT).as_posix(),
        "chunk_file_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "vector_shape": list(vectors.shape),
        "vector_dtype": str(vectors.dtype),
        "normalized": True,
        "row_chunk_ids": chunk_ids,
        "vectors_sha256": hashlib.sha256(
            vector_path.read_bytes()
        ).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
    }

    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    # Verify that disk serialization preserves the vectors.
    restored = np.load(vector_path, allow_pickle=False)
    assert np.array_equal(vectors, restored)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "index_directory": index_dir.relative_to(ROOT).as_posix(),
        "vector_shape": list(vectors.shape),
        "vector_dtype": str(vectors.dtype),
        "vector_array_bytes": vectors.nbytes,
        "disk_roundtrip_exact": True,
    }

    (results_dir / "vector_index_build.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nVector shape:", list(vectors.shape))
    print("Data type:", vectors.dtype)
    print("Vector array bytes:", vectors.nbytes)
    print("Disk roundtrip exact: True")
    print("Index saved to:", index_dir)


if __name__ == "__main__":
    main()