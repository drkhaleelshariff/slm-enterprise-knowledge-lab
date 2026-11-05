import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    session_path = (
        ROOT / "results" / "live_rag_20261105T194249421667Z.json"
    )

    session = json.loads(
        session_path.read_text(encoding="utf-8")
    )

    chunk_path = (
        ROOT / "data" / "knowledge_chunks" / "linux_commands.jsonl"
    )

    current_chunk_hash = hashlib.sha256(
        chunk_path.read_bytes()
    ).hexdigest()

    if current_chunk_hash != session["chunk_file_sha256"]:
        raise ValueError("Chunk file differs from the session's version")

    chunks = {
        chunk["chunk_id"]: chunk
        for line in chunk_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in [json.loads(line)]
    }

    reviews = []

    for record in session["records"]:
        print("\nQuestion:", record["question"])

        if not record["model_called"]:
            print("Result: application withheld; no generated answer to check")

            reviews.append({
                "question": record["question"],
                "status": "application_withheld",
            })
            continue

        evidence = record["supplied_evidence"]
        chunk = chunks[evidence["chunk_id"]]

        # Check that reported provenance matches the indexed chunk.
        for field in (
            "source_path",
            "source_sha256",
            "start_line",
            "end_line",
        ):
            if evidence[field] != chunk[field]:
                raise ValueError(f"Evidence metadata mismatch: {field}")

        source_path = ROOT / chunk["source_path"]
        source_bytes = source_path.read_bytes()

        if hashlib.sha256(source_bytes).hexdigest() != chunk["source_sha256"]:
            raise ValueError("Original source has changed")

        source_lines = source_bytes.decode("utf-8-sig").splitlines()

        section = "\n".join(
            source_lines[chunk["start_line"] - 1:chunk["end_line"]]
        )

        commands = [
            line.removeprefix("Command: ").strip()
            for line in section.splitlines()
            if line.startswith("Command: ")
        ]

        if len(commands) != 1:
            raise ValueError("Expected exactly one source command")

        source_command = commands[0]
        matches = record["answer"].strip() == source_command

        print("Generated answer:", repr(record["answer"]))
        print("Source command:", source_command)
        print("Exact source-command match:", matches)

        reviews.append({
            "question": record["question"],
            "status": "checked",
            "answer": record["answer"],
            "source_command": source_command,
            "exact_source_command_match": matches,
            "source_reference": evidence,
        })

    checked = [
        review for review in reviews
        if review["status"] == "checked"
    ]

    report = {
        "session_file": session_path.name,
        "verification_scope": (
            "Exact command matching against the referenced source lines; "
            "does not establish relevance or source correctness."
        ),
        "generated_answers_checked": len(checked),
        "exact_source_command_matches": sum(
            review["exact_source_command_match"]
            for review in checked
        ),
        "reviews": reviews,
    }

    output_path = ROOT / "results" / "live_answers_verified.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "\nSource-command matches:",
        f"{report['exact_source_command_matches']}/{len(checked)}",
    )
    print("Report saved to:", output_path)


if __name__ == "__main__":
    main()