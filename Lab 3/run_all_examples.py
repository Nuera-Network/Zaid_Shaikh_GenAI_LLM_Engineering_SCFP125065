"""Run all six Lab 3 exercises with fictional, non-sensitive sample data."""

import json

from exercise1_doctor_summary import doctor_summary
from exercise2_review_classifier import classify_review
from exercise3_resume_extractor import extract_resume_fields
from exercise4_support_reply import support_reply
from exercise5_translator import translate_sentence
from exercise6_iterative_refiner import refine_content


def main() -> None:
    print("\nEXERCISE 1 — DOCTOR SUMMARY")
    print(doctor_summary(
        "Dry cough for 3 days, mild sore throat, no fever. No confirmed "
        "diagnosis documented. Clinician recommended rest and fluids.",
        "Asha Patil",
    ))

    print("\nEXERCISE 2 — REVIEW CLASSIFIER")
    print(classify_review("Delivery was fast, but the packaging was damaged."))

    print("\nEXERCISE 3 — RESUME FIELD EXTRACTOR")
    resume = "Sara Khan | sara@example.com | Pune | Skills: Python, SQL"
    print(json.dumps(
        extract_resume_fields(resume, ["Name", "Email", "Skills", "Phone"]),
        indent=2,
        ensure_ascii=False,
    ))

    print("\nEXERCISE 4 — CUSTOMER SUPPORT REPLY")
    print(support_reply(
        "My order arrived two days late. What should I do?",
        "BrightCart",
        35,
    ))

    print("\nEXERCISE 5 — MULTI-LANGUAGE TRANSLATOR")
    print(translate_sentence("Could you please help me?", "Hindi", "formal"))

    print("\nEXERCISE 6 — ITERATIVE CONTENT REFINER")
    draft = "Our team they works very hard and gives best result for customers."
    print("Original:", draft)
    for item in refine_content(draft, "grammar", rounds=3):
        print(f"Round {item['round']}: {item['text']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Network, auth, or provider errors should be visible during a lab run.
        print(f"\nLab run stopped: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

