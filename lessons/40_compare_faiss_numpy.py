import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from saved_index import load_verified_index


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)

    vectors, chunks, manifest = load_verified_index(ROOT)

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    # Exact inner-product search. No index training is required.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print("FAISS version:", faiss.__version__)
    print("Vector dimensions:", index.d)
    print("Stored vectors:", index.ntotal)
    print("Index ready:", index.is_trained)

    configuration = manifest["embedding_model"]

    model = SentenceTransformer(
        configuration["model_id"],
        revision=configuration["revision"],
        device="cpu",
    )

    questions = [
        "Which Linux command prints the kernel release?",
        "Show filesystem space usage.",
        "In a Linux terminal, how do I find my current folder's path?",
        "Meridian escalation telephone?",
    ]

    query_vectors = model.encode(
        questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_vectors = np.ascontiguousarray(
        query_vectors,
        dtype=np.float32,
    )

    # Compare all five candidates so no score is left unchecked.
    faiss_scores, faiss_ids = index.search(
        query_vectors,
        len(chunks),
    )

    numpy_scores = query_vectors @ vectors.T

    records = []

    for query_number, question in enumerate(questions):
        ids = faiss_ids[query_number]

        # Align NumPy scores to FAISS's returned order.
        aligned_numpy_scores = numpy_scores[query_number, ids]

        scores_match = np.allclose(
            faiss_scores[query_number],
            aligned_numpy_scores,
            atol=1e-6,
            rtol=1e-5,
        )

        numpy_ids = np.argsort(
            -numpy_scores[query_number],
            kind="stable",
        )

        ranking_matches = np.array_equal(ids, numpy_ids)

        maximum_difference = float(np.max(np.abs(
            faiss_scores[query_number] - aligned_numpy_scores
        )))

        assert scores_match, "FAISS and NumPy scores differ"

        print(f"\nQuestion: {question}")
        print("Scores agree:", scores_match)
        print("Full ranking agrees:", ranking_matches)
        print(f"Maximum score difference: {maximum_difference:.10f}")

        hits = []

        for rank in range(min(2, len(chunks))):
            row_id = int(ids[rank])
            chunk = chunks[row_id]
            score = float(faiss_scores[query_number, rank])

            print(
                f"  {rank + 1}. {chunk['section_title']} "
                f"| similarity={score:.4f}"
            )

            hits.append({
                "chunk_id": chunk["chunk_id"],
                "section_title": chunk["section_title"],
                "cosine_similarity": score,
            })

        records.append({
            "question": question,
            "scores_agree": bool(scores_match),
            "full_ranking_agrees": bool(ranking_matches),
            "maximum_score_difference": maximum_difference,
            "top_hits": hits,
        })

    report = {
        "faiss_version": faiss.__version__,
        "index_type": "IndexFlatIP",
        "search_type": "exact inner product on normalized vectors",
        "vector_shape": list(vectors.shape),
        "embedding_model": configuration,
        "document_embeddings_recomputed": False,
        "records": records,
        "limitations": [
            "Five chunks only; no speed or scaling claim.",
            "Equal or nearly equal scores can produce different tie ordering.",
            "Search agreement does not establish relevance.",
            "Missing-answer queries still receive nearest neighbours.",
        ],
    }

    output_path = ROOT / "results" / "faiss_numpy_comparison.json"

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()