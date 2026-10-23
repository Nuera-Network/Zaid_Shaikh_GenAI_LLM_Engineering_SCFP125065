"""Offline, deterministic tests; no API key or paid requests are needed."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exercise1_doctor_summary import doctor_summary  # noqa: E402
from exercise2_review_classifier import classify_review  # noqa: E402
from exercise3_resume_extractor import extract_resume_fields  # noqa: E402
from exercise4_support_reply import support_reply, count_words  # noqa: E402
from exercise5_translator import translate_sentence  # noqa: E402
from exercise6_iterative_refiner import refine_content  # noqa: E402


class FakeGroqClient:
    """Mimics the small part of the Groq SDK used in these exercises."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))]
        )


class Lab3Tests(unittest.TestCase):
    def test_doctor_fixed_sections_and_no_assumed_diagnosis(self):
        fake = FakeGroqClient([json.dumps({
            "Symptoms": "Cough for three days",
            "Diagnosis": "Not documented",
            "Recommendation": "Rest and fluids",
        })])
        output = doctor_summary("Cough, advised rest", "Asha", client=fake)
        self.assertEqual(output, (
            "Patient: Asha\n\nSymptoms: Cough for three days\n"
            "Diagnosis: Not documented\nRecommendation: Rest and fluids"
        ))
        self.assertEqual(fake.calls[0]["response_format"]["type"], "json_schema")

    def test_review_output_is_one_label(self):
        fake = FakeGroqClient(['{"label":"Mixed"}'])
        self.assertEqual(classify_review("Good but slow", client=fake), "Mixed")

    def test_review_rejects_unknown_label(self):
        fake = FakeGroqClient(['{"label":"Unclear"}'])
        with self.assertRaises(ValueError):
            classify_review("Review", client=fake)

    def test_resume_exact_fields_missing_as_null(self):
        fake = FakeGroqClient(['{"Name":"Sara","Phone":null}'])
        result = extract_resume_fields("Sara", ["Name", "Phone"], client=fake)
        self.assertEqual(result, {"Name": "Sara", "Phone": None})
        self.assertEqual(
            fake.calls[0]["response_format"]["json_schema"]["schema"]["required"],
            ["Name", "Phone"],
        )

    def test_resume_rejects_unrequested_fields(self):
        fake = FakeGroqClient(['{"Name":"Sara","Phone":null,"Age":"24"}'])
        with self.assertRaises(ValueError):
            extract_resume_fields("Sara", ["Name", "Phone"], client=fake)

    def test_support_word_limit(self):
        fake = FakeGroqClient(["Thank you for contacting BrightCart."])
        reply = support_reply("Delay", "BrightCart", 5, client=fake)
        self.assertLessEqual(count_words(reply), 5)

    def test_support_fallback_when_model_is_too_long(self):
        fake = FakeGroqClient(["word " * 30] * 3)
        reply = support_reply("Late order", "BrightCart", 8, client=fake)
        self.assertLessEqual(count_words(reply), 8)
        self.assertEqual(len(fake.calls), 3)

    def test_translation_formality(self):
        fake = FakeGroqClient(["कृपया मेरी सहायता करें।"])
        result = translate_sentence("Please help me", "Hindi", "formal", client=fake)
        self.assertEqual(result, "कृपया मेरी सहायता करें।")
        self.assertIn("formal", fake.calls[0]["messages"][1]["content"])

    def test_translation_invalid_formality(self):
        with self.assertRaises(ValueError):
            translate_sentence("Hello", "Hindi", "extremely", client=FakeGroqClient([]))

    def test_refiner_has_three_sequential_rounds(self):
        fake = FakeGroqClient(["Draft one.", "Draft two.", "Draft three."])
        result = refine_content("Draft zero.", "grammar", client=fake)
        self.assertEqual([r["round"] for r in result], [1, 2, 3])
        self.assertEqual(result[-1]["text"], "Draft three.")
        self.assertIn("Draft one.", fake.calls[1]["messages"][1]["content"])
        self.assertIn("Draft two.", fake.calls[2]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()

