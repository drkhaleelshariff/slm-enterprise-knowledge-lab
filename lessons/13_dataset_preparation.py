import os

os.environ["HF_HOME"] = r"E:\slm-cache\huggingface"

import json
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict, load_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Load the cleaned JSONL produced in Lesson 1.
    source_path = ROOT / "data" / "sample_clean.jsonl"

    source = load_dataset(
        "json",
        data_files=str(source_path),
        split="train",
    )

    print("1. Existing clean dataset")
    print(source)
    print("Features:", source.features)
    print("First example:", source[0])

    # map() applies a transformation to every record.
    def add_lengths(record):
        return {
            "instruction_characters": len(record["instruction"]),
            "output_characters": len(record["output"]),
        }

    inspected = source.map(add_lengths)

    # filter() keeps only examples meeting a condition.
    nonempty = inspected.filter(
        lambda record: record["output_characters"] > 0
    )

    print("\n2. After map and filter")
    print(nonempty)
    print("First transformed example:", nonempty[0])

    # These are small, explicitly grouped teaching examples.
    # They are separate from our earlier development questions.
    records = [
        {
            "id": "train-pwd-01",
            "group": "linux_current_directory",
            "instruction": "Which Linux command prints the working directory?",
            "input": "",
            "output": "pwd",
        },
        {
            "id": "train-pwd-02",
            "group": "linux_current_directory",
            "instruction": "How can I display my current directory on Linux?",
            "input": "",
            "output": "pwd",
        },
        {
            "id": "train-user-01",
            "group": "linux_current_user",
            "instruction": "Which command prints my current Linux user name?",
            "input": "",
            "output": "whoami",
        },
        {
            "id": "train-user-02",
            "group": "linux_current_user",
            "instruction": "How do I identify the current user on Linux?",
            "input": "",
            "output": "whoami",
        },
        {
            "id": "validation-host-01",
            "group": "linux_hostname",
            "instruction": "Which command displays the Linux host name?",
            "input": "",
            "output": "hostname",
        },
        {
            "id": "validation-host-02",
            "group": "linux_hostname",
            "instruction": "How do I print the machine host name on Linux?",
            "input": "",
            "output": "hostname",
        },
    ]

    teaching_dataset = Dataset.from_list(records)

    # Keep paraphrases of the same underlying task together.
    train_groups = {
        "linux_current_directory",
        "linux_current_user",
    }
    validation_groups = {"linux_hostname"}

    splits = DatasetDict({
        "train": teaching_dataset.filter(
            lambda record: record["group"] in train_groups
        ),
        "validation": teaching_dataset.filter(
            lambda record: record["group"] in validation_groups
        ),
    })

    actual_train_groups = set(splits["train"]["group"])
    actual_validation_groups = set(splits["validation"]["group"])

    overlap = actual_train_groups & actual_validation_groups

    assert not overlap, f"Group leakage detected: {overlap}"

    all_ids = (
        list(splits["train"]["id"])
        + list(splits["validation"]["id"])
    )
    assert len(all_ids) == len(set(all_ids)), "Duplicate example IDs"

    print("\n3. Group-separated teaching splits")
    print(splits)
    print("Train groups:", sorted(actual_train_groups))
    print("Validation groups:", sorted(actual_validation_groups))
    print("Group overlap:", sorted(overlap))

    destination = ROOT / "data" / "teaching_splits"
    destination.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        split.to_json(
            str(destination / f"{name}.jsonl"),
            orient="records",
            lines=True,
            force_ascii=False,
        )

    report = {
        "datasets_version": datasets.__version__,
        "source_rows": len(source),
        "rows_after_filter": len(nonempty),
        "split_rows": {
            name: len(split)
            for name, split in splits.items()
        },
        "train_groups": sorted(actual_train_groups),
        "validation_groups": sorted(actual_validation_groups),
        "group_overlap": sorted(overlap),
        "status": "teaching example, not a training-ready corpus",
        "limitations": [
            "Only six examples.",
            "Validation covers one task group.",
            "Examples are visible development material.",
            "No sealed test set has been created.",
        ],
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "dataset_preparation.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSplits saved to: {destination}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()