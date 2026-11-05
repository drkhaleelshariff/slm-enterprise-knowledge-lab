import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source_path = ROOT / "results" / "semantic_retrieval.json"

    retrieval = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    threshold = 0.50
    records = []

    for record in retrieval["records"]:
        hits = record["hits"]
        top_hit = hits[0] if hits else None

        score = (
            top_hit["cosine_similarity"]
            if top_hit is not None
            else None
        )

        passes_gate = (
            score is not None and score >= threshold
        )

        print(f"\nQuestion: {record['query']}")
        print("Top similarity:", score)

        if passes_gate:
            print("Decision: candidate passes retrieval gate")
            print("Section:", top_hit["section_title"])
            print(
                "Source:",
                f"{top_hit['source_path']}:"
                f"{top_hit['start_line']}-{top_hit['end_line']}",
            )
        else:
            print("Decision: withhold evidence from generation")

        records.append({
            "query": record["query"],
            "top_similarity": score,
            "passes_gate": passes_gate,
            "selected_chunk": top_hit if passes_gate else None,
        })

    report = {
        "source_report": source_path.relative_to(ROOT).as_posix(),
        "threshold": threshold,
        "threshold_status": (
            "Teaching threshold selected after inspecting these examples; "
            "not independently validated."
        ),
        "records": records,
        "limitations": [
            "Passing the gate does not prove answer sufficiency.",
            "Failing the gate does not prove the answer is absent.",
            "Threshold behaviour depends on the model, corpus, and queries.",
            "Ambiguous queries may need clarification.",
            "This script reviews saved scores; it does not run a live search.",
        ],
    }

    output_path = ROOT / "results" / "retrieval_gate.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()