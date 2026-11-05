import hashlib
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

REQUIRED_FIELDS = ("id", "instruction", "input", "output")


def normalise_text(text: str) -> str:
    """Collapse repeated whitespace in prose."""
    return " ".join(text.split())


def validate_record(record: object) -> dict[str, str]:
    """Validate a record and return a cleaned copy."""
    if not isinstance(record, dict):
        raise ValueError("Record must be a JSON object")

    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"Missing field: {field}")

        if not isinstance(record[field], str):
            raise ValueError(f"{field} must be a string")

    cleaned = {
        field: normalise_text(record[field])
        for field in REQUIRED_FIELDS
    }

    for field in ("id", "instruction", "output"):
        if not cleaned[field]:
            raise ValueError(f"{field} must not be empty")

    return cleaned


def content_fingerprint(record: dict[str, str]) -> str:
    """Identify duplicate content, ignoring the record ID."""
    content = {
        field: record[field]
        for field in ("instruction", "input", "output")
    }

    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path):
    """Read one JSON record at a time."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "_parse_error": f"{exc.msg}, column {exc.colno}"
                }


def create_sample_data(path: Path) -> None:
    """Create two valid examples, one duplicate, and one invalid example."""
    records = [
        {
            "id": "ex-001",
            "instruction": "How do I check disk space on Linux?",
            "input": "",
            "output": (
                "Run df -h to inspect filesystem capacity "
                "and available space."
            ),
        },
        {
            "id": "ex-002",
            "instruction": "How do I list running processes on Linux?",
            "input": "",
            "output": (
                "Run ps aux to list processes, "
                "or top for an interactive view."
            ),
        },
        {
            "id": "ex-003",
            "instruction": "How do I check disk space on Linux?",
            "input": "",
            "output": (
                "Run df -h to inspect filesystem capacity "
                "and available space."
            ),
        },
        {
            "id": "ex-004",
            "instruction": "How do I check my Python version?",
            "input": "",
            "output": "   ",
        },
    ]

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "sample_raw.jsonl"
    clean_path = DATA_DIR / "sample_clean.jsonl"

    # Preserve any changes you make to the raw dataset.
    if not raw_path.exists():
        create_sample_data(raw_path)

    seen_content = set()
    seen_ids = set()

    counts = {
        "accepted": 0,
        "duplicates": 0,
        "invalid": 0,
    }

    with clean_path.open("w", encoding="utf-8") as destination:
        for line_number, raw_record in read_jsonl(raw_path):
            try:
                if (
                    isinstance(raw_record, dict)
                    and "_parse_error" in raw_record
                ):
                    raise ValueError(raw_record["_parse_error"])

                record = validate_record(raw_record)

                if record["id"] in seen_ids:
                    raise ValueError(
                        f"Repeated ID: {record['id']}"
                    )

                seen_ids.add(record["id"])
                fingerprint = content_fingerprint(record)

                if fingerprint in seen_content:
                    counts["duplicates"] += 1
                    logging.warning(
                        "Line %s: duplicate content skipped",
                        line_number,
                    )
                    continue

                seen_content.add(fingerprint)

                destination.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                counts["accepted"] += 1

            except ValueError as exc:
                counts["invalid"] += 1
                logging.warning(
                    "Line %s: %s",
                    line_number,
                    exc,
                )

    report_path = RESULTS_DIR / "dataset_quality.json"

    report_path.write_text(
        json.dumps(counts, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nDataset quality summary:")
    print(json.dumps(counts, indent=2))

    print(f"\nClean dataset: {clean_path}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    main()