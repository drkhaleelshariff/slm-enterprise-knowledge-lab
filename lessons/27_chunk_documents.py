import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source_path = ROOT / "data" / "knowledge_sources" / "linux_commands.md"

    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    # utf-8-sig also handles files saved with a UTF-8 BOM.
    text = source_bytes.decode("utf-8-sig")
    lines = text.splitlines()

    # This parser targets our simple reference-document format.
    metadata = {}

    for line in lines:
        if line.startswith("## "):
            break

        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key.strip()] = value.strip()

    document_id = metadata.get("Document ID")
    document_version = metadata.get("Document version")

    if not document_id or not document_version:
        raise ValueError("Document ID and version are required")

    section_starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("## ")
    ]

    if not section_starts:
        raise ValueError("No level-two sections found")

    chunks = []

    for position, start in enumerate(section_starts):
        end = (
            section_starts[position + 1]
            if position + 1 < len(section_starts)
            else len(lines)
        )

        # Exclude trailing blank lines from the source range.
        while end > start and not lines[end - 1].strip():
            end -= 1

        title = lines[start][3:].strip()
        body = "\n".join(lines[start + 1:end]).strip()

        if not title or not body:
            raise ValueError("Empty section title or body")

        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            title.lower(),
        ).strip("-")

        chunk_id = f"{document_id}:v{document_version}:{slug}"

        chunks.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "document_version": document_version,
            "source_path": source_path.relative_to(ROOT).as_posix(),
            "source_sha256": source_hash,
            "section_title": title,
            "start_line": start + 1,
            "end_line": end,
            "text": f"{title}\n\n{body}",
        })

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs")

    output_dir = ROOT / "data" / "knowledge_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "linux_commands.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    for chunk in chunks:
        print(f"\nChunk: {chunk['chunk_id']}")
        print(
            f"Source lines: "
            f"{chunk['start_line']}-{chunk['end_line']}"
        )
        print(chunk["text"])

    report = {
        "document_id": document_id,
        "document_version": document_version,
        "source_sha256": source_hash,
        "chunking_strategy": "one level-two section per chunk",
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "document_chunking.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {len(chunks)}")
    print("Chunks saved to:", output_path)
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()