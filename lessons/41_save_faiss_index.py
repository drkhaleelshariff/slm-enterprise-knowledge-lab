import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from saved_index import load_verified_index


ROOT = Path(__file__).resolve().parents[1]


def main():
    faiss.omp_set_num_threads(2)

    vectors, chunks, source_manifest = load_verified_index(ROOT)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_dir = ROOT / "indexes" / "linux_commands_faiss_v1"

    if output_dir.exists():
        raise FileExistsError(
            "FAISS output folder already exists. "
            "Choose a new folder name for another build."
        )

    output_dir.mkdir(parents=True)

    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "chunks.json"

    faiss.write_index(index, str(index_path))

    metadata_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "normalized_vectors": True,
        "dimension": index.d,
        "vector_count": index.ntotal,
        "embedding_model": source_manifest["embedding_model"],
        "chunk_file": source_manifest["chunk_file"],
        "chunk_file_sha256": source_manifest["chunk_file_sha256"],
        "row_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "faiss_version": faiss.__version__,
        "index_sha256": hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    # Reload our locally generated index.
    restored = faiss.read_index(str(index_path))

    assert restored.d == index.d
    assert restored.ntotal == index.ntotal
    assert restored.metric_type == faiss.METRIC_INNER_PRODUCT

    # Use all document vectors as test queries.
    original_scores, original_ids = index.search(vectors, len(chunks))
    restored_scores, restored_ids = restored.search(vectors, len(chunks))

    same_ids = np.array_equal(original_ids, restored_ids)
    same_scores = np.array_equal(original_scores, restored_scores)

    assert same_ids, "Rankings changed after reloading"
    assert same_scores, "Scores changed after reloading"

    report = {
        "index_directory": output_dir.relative_to(ROOT).as_posix(),
        "index_type": "IndexFlatIP",
        "dimension": restored.d,
        "vector_count": restored.ntotal,
        "roundtrip_rankings_identical": bool(same_ids),
        "roundtrip_scores_identical": bool(same_scores),
        "test_queries": "all five document vectors",
        "index_file_bytes": index_path.stat().st_size,
        "document_embeddings_recomputed": False,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "faiss_index_build.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved index type:", manifest["index_type"])
    print("Dimensions:", restored.d)
    print("Stored vectors:", restored.ntotal)
    print("Reloaded rankings identical:", same_ids)
    print("Reloaded scores identical:", same_scores)
    print("Index saved to:", output_dir)
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()