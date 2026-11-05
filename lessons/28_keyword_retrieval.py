import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STOP_WORDS = {
    "a", "an", "the", "is", "of", "on", "in", "to",
    "which", "what", "how", "can", "i", "my", "this",
    "does", "do", "and", "for", "with",
}


def terms(text):
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    }


def main():
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

    # Refuse to search chunks generated from outdated source files.
    for chunk in chunks:
        source_path = ROOT / chunk["source_path"]
        current_hash = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

        if current_hash != chunk["source_sha256"]:
            raise ValueError(
                "Source changed. Rerun Lesson 27 before searching."
            )

    def retrieve(query, top_k=2):
        query_terms = terms(query)
        candidates = []

        for chunk in chunks:
            matched = query_terms & terms(chunk["text"])

            # Simple count of distinct overlapping terms.
            score = len(matched)

            # Avoid returning zero-overlap chunks.
            if score > 0:
                candidates.append({
                    **chunk,
                    "score": score,
                    "matched_terms": sorted(matched),
                })

        return sorted(
            candidates,
            key=lambda item: (-item["score"], item["chunk_id"]),
        )[:top_k]

    cases = [
        {
            "id": "kernel",
            "query": "Which Linux command prints the kernel release?",
            "expected_chunk": "linux-commands-001:v1:kernel-release",
        },
        {
            "id": "space",
            "query": "Show filesystem space usage.",
            "expected_chunk": "linux-commands-001:v1:filesystem-space",
        },
        {
            "id": "paraphrase",
            "query": "Where am I?",
            "expected_chunk": (
                "linux-commands-001:v1:current-working-directory"
            ),
        },
        {
            "id": "missing-policy",
            "query": "Meridian escalation telephone?",
            "expected_chunk": None,
        },
    ]

    records = []

    for case in cases:
        hits = retrieve(case["query"])

        print(f"\nQuestion: {case['query']}")

        if not hits:
            print("No keyword matches.")
        else:
            for rank, hit in enumerate(hits, start=1):
                print(
                    f"{rank}. {hit['section_title']} | "
                    f"score={hit['score']} | "
                    f"matched={hit['matched_terms']}"
                )
                print(
                    f"   Source: {hit['source_path']}:"
                    f"{hit['start_line']}-{hit['end_line']}"
                )

        expected = case["expected_chunk"]

        top1_correct = (
            hits[0]["chunk_id"] == expected
            if expected is not None and hits
            else False if expected is not None
            else None
        )

        records.append({
            **case,
            "top1_correct": top1_correct,
            "hits": hits,
        })

    report = {
        "method": "distinct keyword overlap with stop-word removal",
        "top_k": 2,
        "chunk_count": len(chunks),
        "records": records,
        "limitations": [
            "No semantic understanding or synonym handling.",
            "Shared generic words can produce irrelevant matches.",
            "A positive score does not establish answer sufficiency.",
            "No keyword match does not prove the source lacks an answer.",
        ],
    }

    output_path = ROOT / "results" / "keyword_retrieval.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()