import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source_path = ROOT / "results" / "pretrained_baseline.json"
    report = json.loads(source_path.read_text(encoding="utf-8"))

    # Explicit review of the outputs observed in this run.
    # This is not an automated correctness evaluator.
    reviews = {
        "linux-completion": {
            "passed": False,
            "category": "incorrect_command",
            "reason": (
                "The continuation suggests sudo loskill -s /dev/sda "
                "instead of a filesystem-space command."
            ),
            "expected_behaviour": "Suggest df or df -h.",
        },
        "python-completion": {
            "passed": False,
            "category": "incorrect_command_and_explanation",
            "reason": (
                "Installing requirements does not display the Python "
                "version and does not install the Python interpreter."
            ),
            "expected_behaviour": (
                "Suggest python --version, python -V, "
                "or an appropriate equivalent."
            ),
        },
        "unknown-policy": {
            "passed": False,
            "category": "unsupported_claim",
            "reason": (
                "The continuation supplies a phone number despite "
                "having no evidence for the fictional company's policy."
            ),
            "expected_behaviour": (
                "State that the number is unknown and request "
                "an authoritative policy or directory."
            ),
        },
    }

    for case in report["cases"]:
        case["human_review"] = reviews[case["id"]]

    passed = sum(
        case["human_review"]["passed"]
        for case in report["cases"]
    )

    report["review_summary"] = {
        "method": "explicit review of this observed run",
        "passed": passed,
        "total": len(report["cases"]),
        "limitation": (
            "Three illustrative cases, not a general accuracy estimate. "
            "Review must be revisited if generated outputs change."
        ),
    }

    output_path = ROOT / "results" / "pretrained_baseline_reviewed.json"

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Passed: {passed}/{len(report['cases'])}")

    for case in report["cases"]:
        review = case["human_review"]
        print(f"{case['id']}: {review['category']}")

    print(f"\nReviewed report saved to: {output_path}")


if __name__ == "__main__":
    main()