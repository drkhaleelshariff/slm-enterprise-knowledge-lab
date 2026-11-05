import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source_path = ROOT / "results" / "irrelevant_context_control.json"

    report = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    uncertainty_response = (
        "I do not know from the supplied information."
    )

    reviewed = []

    for record in report["records"]:
        if record["route"] != "generation_comparison":
            continue

        answer = record["with_context"]["answer"]

        review = {
            "query": record["query"],
            "answer": answer,
            "reference_command_match": record["with_context_exact_match"],
            "explicit_uncertainty_match": answer == uncertainty_response,
            "supplied_context_contains_answer": False,
            "assessment": (
                "Appropriate withholding under an evidence-only policy."
                if answer == uncertainty_response
                else "Requires manual review."
            ),
        }

        reviewed.append(review)

        print("\nQuestion:", review["query"])
        print("Answer:", repr(answer))
        print("Assessment:", review["assessment"])

    output = {
        "source_report": source_path.relative_to(ROOT).as_posix(),
        "review_policy": (
            "When supplied context lacks supporting evidence, "
            "withhold the answer explicitly."
        ),
        "review_basis": (
            "Manual inspection of the office-supplies passage "
            "and the three Linux questions."
        ),
        "reviewed_model_responses": len(reviewed),
        "explicit_uncertainty_matches": sum(
            item["explicit_uncertainty_match"]
            for item in reviewed
        ),
        "records": reviewed,
        "limitations": [
            "One irrelevant passage used for three questions.",
            "No contradictory or malicious-context tests.",
            "The original prompt did not explicitly require "
            "exclusive reliance on supplied context.",
            "Exact phrase matching is narrower than semantic review.",
        ],
    }

    output_path = (
        ROOT / "results" / "irrelevant_context_control_reviewed.json"
    )

    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "\nExplicit uncertainty responses:",
        f"{output['explicit_uncertainty_matches']}/{len(reviewed)}",
    )
    print("Review saved to:", output_path)


if __name__ == "__main__":
    main()