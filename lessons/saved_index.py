import hashlib
import json
from pathlib import Path

import numpy as np


def load_verified_index(root: Path, index_name="linux_commands_v1"):
    index_dir = root / "indexes" / index_name

    manifest = json.loads(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )

    for filename, hash_field in [
        ("vectors.npy", "vectors_sha256"),
        ("chunks.json", "metadata_sha256"),
    ]:
        actual_hash = hashlib.sha256(
            (index_dir / filename).read_bytes()
        ).hexdigest()

        if actual_hash != manifest[hash_field]:
            raise ValueError(f"{filename} differs from the manifest")

    chunk_path = root / manifest["chunk_file"]

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

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a nonempty matrix")

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
            (root / chunk["source_path"]).read_bytes()
        ).hexdigest()

        if source_hash != chunk["source_sha256"]:
            raise ValueError("Source changed. Rebuild chunks and index.")

    return vectors, chunks, manifest