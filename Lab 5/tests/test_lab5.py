"""Deterministic, offline integration tests for all six LCEL pipelines."""

import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.runnables import RunnableLambda  # noqa: E402

from exercise1_review_to_ticket import review_to_ticket  # noqa: E402
from exercise2_abstract_to_summary import abstract_to_summary  # noqa: E402
from exercise3_interview_scorecard import interview_to_scorecard  # noqa: E402
from exercise4_event_weather import event_weather_recommendation  # noqa: E402
from exercise5_support_escalation import support_escalation  # noqa: E402
from exercise6_weekly_workout import workout_goal_to_plan  # noqa: E402
from weather_api import WeatherUnavailable, fetch_forecast  # noqa: E402


def fake_llm(*responses):
    items = iter(responses)
    seen_prompts = []

    def reply(prompt):
        seen_prompts.append(prompt.to_string())
        return next(items)

    return RunnableLambda(reply), seen_prompts


class WorkflowTests(unittest.TestCase):
    def test_review_ticket_two_steps(self):
        model, prompts = fake_llm(
            json.dumps({
                "complaint": "Bluetooth drops", "product_or_feature": "Headphones",
                "sentiment": "negative",
            }),
            "Subject: Bluetooth drops\nDetails: Headphones connection is unstable.",
        )
        result = review_to_ticket("SECRET_RAW_TAG Headphones connection drops", llm=model)
        self.assertEqual(result["step1_complaint_data"]["sentiment"], "negative")
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("SECRET_RAW_TAG", prompts[1])
        self.assertIn("Bluetooth drops", prompts[1])

    def test_abstract_summary_two_steps(self):
        model, prompts = fake_llm(
            json.dumps({
                "research_question": "Do breaks improve recall?",
                "method": "Randomized experiment",
                "key_finding": "Break group recalled more items",
            }),
            "Scientists compared recall after breaks and found better immediate recall.",
        )
        result = abstract_to_summary("RAW_ONLY: students with rest performed better", llm=model)
        self.assertIn("breaks", result["step1_research_data"]["research_question"])
        self.assertNotIn("RAW_ONLY", prompts[1])

    def test_interview_three_steps_and_skill_validation(self):
        model, prompts = fake_llm(
            json.dumps({"answers": [{"question": "What is SQL?", "answer": "I use JOIN."}]}),
            json.dumps({"assessments": [
                {"skill": "SQL", "demonstrated": True, "evidence_or_gap": "Candidate uses JOIN"},
                {"skill": "Python", "demonstrated": False, "evidence_or_gap": "Not discussed"},
            ]}),
            json.dumps({
                "rows": [
                    {"skill": "SQL", "demonstrated": True, "evidence_or_gap": "JOIN demonstrated"},
                    {"skill": "Python", "demonstrated": False, "evidence_or_gap": "Missing evidence"},
                ],
                "overall_recommendation": "Request additional Python evidence",
            }),
        )
        result = interview_to_scorecard("RAW_TOKEN: Q: SQL? A: JOIN", ["SQL", "Python"], llm=model)
        self.assertEqual(len(prompts), 3)
        self.assertNotIn("RAW_TOKEN", prompts[1])
        self.assertNotIn("RAW_TOKEN", prompts[2])
        self.assertFalse(result["step3_hiring_scorecard"]["rows"][1]["demonstrated"])

    def test_event_weather_api_and_three_steps(self):
        event_date = (date.today() + timedelta(days=3)).isoformat()
        model, prompts = fake_llm(
            json.dumps({
                "event_type": "sports day", "event_date": event_date,
                "location": "Pune", "indoor_outdoor_preference": "outdoor",
            }),
            "Move indoors due to thunderstorms in the forecast; recheck nearer the event.",
        )
        queries = []

        def weather(location, requested_date):
            queries.append((location, requested_date))
            return {"temperature": 26.0, "precipitation_chance": 90.0,
                    "conditions": "thunderstorms"}

        result = event_weather_recommendation(
            "ORIGINAL_ONLY: sports day outside in Pune", llm=model, weather_fetcher=weather
        )
        self.assertEqual(queries, [("Pune", event_date)])
        self.assertEqual(result["step2_weather"]["precipitation_chance"], 90.0)
        self.assertEqual(len(prompts), 2)  # Weather API replaces an LLM call.
        self.assertNotIn("ORIGINAL_ONLY", prompts[1])

    def test_escalation_three_steps(self):
        model, prompts = fake_llm(
            json.dumps({"issue": "Login error", "customer_tone": "Frustrated",
                        "resolution_attempted": "Restarted app"}),
            json.dumps({"missing_fields": ["account ID", "error code"],
                        "is_ready_to_escalate": False}),
            "Priority: provisional. Obtain account ID and error code first.",
        )
        result = support_escalation("RAW_CHAT_ONLY: cannot log in", llm=model)
        self.assertEqual(len(prompts), 3)
        self.assertNotIn("RAW_CHAT_ONLY", prompts[1])
        self.assertNotIn("RAW_CHAT_ONLY", prompts[2])
        self.assertFalse(result["step2_escalation_gaps"]["is_ready_to_escalate"])

    def test_workout_plan_matches_days(self):
        model, prompts = fake_llm(
            json.dumps({"goal_type": "stamina", "fitness_level": "beginner",
                        "days_available": 4, "equipment_access": "none"}),
            json.dumps({"days": [
                {"day_number": 1, "activity": "Walk", "duration_minutes": 20, "intensity": "low"},
                {"day_number": 2, "activity": "Mobility", "duration_minutes": 20, "intensity": "low"},
            ]}),
        )
        result = workout_goal_to_plan("RAW_FITNESS_ONLY: stamina at home", 2, llm=model)
        self.assertEqual(len(result["step2_weekly_plan"]["days"]), 2)
        self.assertEqual(result["step1_fitness_goal"]["days_available"], 2)
        self.assertNotIn("RAW_FITNESS_ONLY", prompts[1])

    def test_workout_rejects_wrong_schedule_length(self):
        model, _ = fake_llm(
            json.dumps({"goal_type": "stamina", "fitness_level": "beginner",
                        "days_available": 2, "equipment_access": "none"}),
            json.dumps({"days": [
                {"day_number": 1, "activity": "Walk", "duration_minutes": 20, "intensity": "low"},
            ]}),
        )
        with self.assertRaises(ValueError):
            workout_goal_to_plan("Improve stamina", 2, llm=model)

    @patch("weather_api._get_json")
    def test_real_weather_adapter_normalizes_api_fields(self, mock_get):
        target = (date.today() + timedelta(days=2)).isoformat()
        mock_get.side_effect = [
            {"results": [{"latitude": 18.52, "longitude": 73.85, "timezone": "UTC"}]},
            {"daily": {"time": [target], "temperature_2m_max": [28.5],
                       "precipitation_probability_max": [75], "weather_code": [63]}},
        ]
        result = fetch_forecast("Pune", target)
        self.assertEqual(result, {
            "temperature": 28.5, "precipitation_chance": 75.0,
            "conditions": "rain",
        })
        self.assertEqual(mock_get.call_count, 2)

    @patch("weather_api._get_json")
    def test_weather_out_of_range_is_not_fabricated(self, mock_get):
        mock_get.return_value = {
            "results": [{"latitude": 18.52, "longitude": 73.85, "timezone": "UTC"}]
        }
        target = (date.today() + timedelta(days=40)).isoformat()
        with self.assertRaises(WeatherUnavailable):
            fetch_forecast("Pune", target)
        self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()

