import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "command_assistant_v3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_records(filename):
    path = DATA_DIR / filename
    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{filename}, line {line_number}: {exc}"
            ) from exc

        require(
            isinstance(record, dict),
            f"{filename}, line {line_number}: expected a JSON object",
        )

        for field in ("id", "group", "instruction", "input", "output"):
            require(
                isinstance(record.get(field), str),
                f"{filename}, line {line_number}: "
                f"{field} must be a string",
            )

            if field != "input":
                require(
                    bool(record[field].strip()),
                    f"{filename}, line {line_number}: {field} is empty",
                )

        records.append(record)

    require(bool(records), f"{filename} contains no records")
    return records


def question_key(record):
    # Catch exact matches despite differences in case or whitespace.
    return (
        " ".join(record["instruction"].casefold().split()),
        " ".join(record["input"].casefold().split()),
    )


def main():
    train = load_records("train.jsonl")
    development = load_records("development.jsonl")

    all_records = train + development
    ids = [record["id"] for record in all_records]

    require(
        len(ids) == len(set(ids)),
        "Duplicate IDs found within or across files",
    )

    for name, records in (
        ("training", train),
        ("development", development),
    ):
        keys = [question_key(record) for record in records]
        require(
            len(keys) == len(set(keys)),
            f"Duplicate question/context pairs in {name}",
        )

    train_keys = {question_key(record) for record in train}
    development_keys = {
        question_key(record) for record in development
    }

    require(
        not train_keys.intersection(development_keys),
        "A question/context pair appears in both files",
    )

    train_groups = {record["group"] for record in train}

    allowed_types = {
        "known_task_new_wording",
        "unseen_command_task",
        "new_unknown_fact",
    }

    for record in development:
        evaluation_type = record.get("evaluation_type")

        require(
            evaluation_type in allowed_types,
            f"{record['id']}: invalid evaluation_type",
        )

        if evaluation_type == "known_task_new_wording":
            require(
                record["group"] in train_groups,
                f"{record['id']}: task group is absent from training",
            )
        else:
            require(
                record["group"] not in train_groups,
                f"{record['id']}: supposedly new group appears in training",
            )

    instruction = (
        DATA_DIR / "instruction.txt"
    ).read_text(encoding="utf-8-sig").strip()

    require(bool(instruction), "Shared instruction is empty")

    counts = dict(Counter(
        record["evaluation_type"]
        for record in development
    ))

    report = {
        "training_records": len(train),
        "development_records": len(development),
        "unique_ids_across_files": len(set(ids)),
        "duplicate_question_context_pairs_across_files": 0,
        "development_types": counts,
        "checks_passed": True,
        "limitations": [
            "Checks do not establish factual correctness.",
            "Exact-match checks do not detect semantic paraphrases.",
            "Task groups are manually assigned.",
            "Development data is visible, not a sealed test set.",
        ],
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "v3_data_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Training records:", len(train))
    print("Development records:", len(development))
    print("Unique IDs across files:", len(set(ids)))
    print("Duplicate question/context pairs across files: 0")
    print("Development types:", counts)
    print("\nAll dataset checks passed.")
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()
