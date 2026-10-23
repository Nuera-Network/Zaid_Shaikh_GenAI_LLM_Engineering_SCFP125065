"""Offline Lab 6 verification: no Groq key or live external data required."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_tool_flow, used_tool_names  # noqa: E402
from exercise1_order_lookup import lookup_order_status, run as order_run  # noqa: E402
from exercise2_loan_emi import calculate_loan_emi, emi_tool  # noqa: E402
from exercise3_unit_converter import convert_distance, convert_temperature  # noqa: E402
from exercise4_weather_outfit import (  # noqa: E402
    ensure_weather_provenance, run as outfit_run,
)
from exercise5_async_stock import stock_tool  # noqa: E402
from exercise6_expense_split import expense_tool, run as expense_run  # noqa: E402
from exercise7_trip_cost import TRIP_TOOLS, run as trip_run  # noqa: E402
from exercise8_health_toolkit import HealthToolkit, calculate_bmi, calculate_daily_calories  # noqa: E402
from exercise9_meeting_scheduler import (  # noqa: E402
    DEMO_BOOKINGS, MEETING_TOOLS, booking_guard, run as meeting_run,
)
from exercise10_restaurant_assistant import RESTAURANT_TOOLS, run as restaurant_run  # noqa: E402


def calls(*items):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"id-{index}"}
        for index, (name, args) in enumerate(items)
    ])


class FakeToolCallingLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.seen = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools.append([tool.name for tool in tools])
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return next(self.responses)


class Lab6Tests(unittest.TestCase):
    def test_01_order_lookup_model_call_results_and_final(self):
        llm = FakeToolCallingLLM(
            calls(("lookup_order_status", {"order_id": "ORD-4521"})),
            AIMessage(content="Your sample order is in transit."),
        )
        flow = order_run(llm=llm)
        self.assertEqual(flow["rounds"][0]["manual_results"][0]["result"]["found"], True)
        self.assertEqual(len(flow["rounds"][0]["tool_calls"]), 1)
        self.assertTrue(any(isinstance(m, ToolMessage) for m in llm.seen[1]))

    def test_02_emi_and_strict_validation(self):
        result = emi_tool.invoke({"principal": 500000, "annual_rate": 9, "tenure_months": 24})
        self.assertAlmostEqual(result["monthly_emi_inr"], 22842.37, places=1)
        self.assertAlmostEqual(calculate_loan_emi(1200, 0, 12)["monthly_emi_inr"], 100)
        for data in (
            {"principal": 0, "annual_rate": 9, "tenure_months": 24},
            {"principal": 1000, "annual_rate": 31, "tenure_months": 24},
            {"principal": 1000, "annual_rate": 8, "tenure_months": 0},
        ):
            with self.assertRaises(Exception):
                emi_tool.invoke(data)

    def test_03_two_conversion_tools(self):
        self.assertAlmostEqual(convert_distance.invoke({
            "value": 100, "from_unit": "km", "to_unit": "miles"
        })["result"], 62.1371, places=4)
        self.assertEqual(convert_temperature.invoke({
            "value": 100, "from_unit": "C", "to_unit": "F"
        })["result"], 212.0)

    @patch("exercise4_weather_outfit.current_weather")
    def test_04_weather_chained_without_invented_arguments(self, mocked):
        mocked.return_value = {"city": "Pune", "temperature": 25.0,
                               "conditions": "rainy", "source": "TEST", "observed_at": "now"}
        llm = FakeToolCallingLLM(
            calls(("get_weather", {"city": "Pune"})),
            calls(("recommend_outfit", {"temperature": 25.0, "conditions": "rainy"})),
            AIMessage(content="Wear breathable clothes and take an umbrella."),
        )
        result = outfit_run("What to wear in Pune?", llm=llm)
        self.assertTrue(result["outfit_used_verified_weather"])
        self.assertEqual(used_tool_names(result), ["get_weather", "recommend_outfit"])
        self.assertIsNotNone(ensure_weather_provenance(
            "recommend_outfit", {"temperature": 44, "conditions": "sunny"},
            [{"tool": "get_weather", "result": mocked.return_value, "success": True}],
        ))

    def test_05_stock_sync_async_equal(self):
        sync = stock_tool.invoke({"product_id": "P1023"})
        async_result = asyncio.run(stock_tool.ainvoke({"product_id": "P1023"}))
        self.assertEqual(sync, async_result)
        self.assertTrue(sync["in_stock"])

    def test_06_expense_direct_equals_model_tool(self):
        llm = FakeToolCallingLLM(
            calls(("split_expense", {"total_amount": 2400, "num_people": 4})),
            AIMessage(content="Each friend owes ₹600."),
        )
        result = expense_run(llm=llm)
        self.assertTrue(result["manual_matches_bound_result"])
        self.assertEqual(result["manual_result"]["per_person_inr"], "600.00")
        with self.assertRaises(Exception):
            expense_tool.invoke({"total_amount": 2400, "num_people": 0})

    def test_07_three_travel_tools_one_query(self):
        llm = FakeToolCallingLLM(
            calls(
                ("get_flight_cost", {"origin": "Delhi", "destination": "Goa"}),
                ("get_hotel_cost", {"city": "Goa", "nights": 3}),
                ("get_food_budget", {"city": "Goa", "days": 4}),
            ),
            AIMessage(content="These are only sample budgets."),
        )
        result = trip_run(llm=llm)
        self.assertTrue(result["all_three_called"])
        self.assertEqual(result["python_calculated_total_inr"], 8200 + 9600 + 4400)
        self.assertEqual(len(llm.bound_tools[0]), len(TRIP_TOOLS))

    def test_08_health_toolkit_both_structured(self):
        tools = HealthToolkit().get_tools()
        self.assertEqual({tool.name for tool in tools}, {"calculate_bmi", "calculate_daily_calories"})
        self.assertAlmostEqual(calculate_bmi(70, 175)["bmi"], 22.86, places=2)
        calories = calculate_daily_calories(70, 175, 25, "moderate")
        self.assertLess(*calories["estimated_daily_maintenance_kcal_range"])

    def test_09_booking_needs_checks_and_is_in_memory(self):
        DEMO_BOOKINGS.clear()
        from datetime import date, timedelta
        when = (date.today() + timedelta(days=2)).isoformat()
        llm = FakeToolCallingLLM(
            calls(("check_calendar_availability", {"person": "Zaid", "date": when})),
            AIMessage(content="Sample availability checked."),
            calls(
                ("check_calendar_availability", {"person": "Zaid", "date": when}),
                ("get_timezone_offset", {"city": "Pune"}),
            ),
            calls(("book_meeting", {"person": "Zaid", "date": when, "time": "15:00"})),
            AIMessage(content="Sample meeting booked."),
        )
        result = meeting_run(llm=llm)
        self.assertEqual(result["availability_tool_names"], ["check_calendar_availability"])
        self.assertIn("book_meeting", result["booking_tool_names"])
        self.assertEqual(len(DEMO_BOOKINGS), 1)
        DEMO_BOOKINGS.clear()
        self.assertIsNotNone(booking_guard("book_meeting", {"person": "Zaid"}, []))

    def test_10_restaurant_four_tools_bound(self):
        llm = FakeToolCallingLLM(
            calls(("find_restaurants", {"cuisine": "Italian", "city": "Pune"})),
            calls(
                ("get_restaurant_rating", {"restaurant_name": "Demo Trattoria"}),
                ("check_restaurant_open", {"restaurant_name": "Demo Trattoria", "day": "Friday"}),
                ("estimate_cost_for_two", {"restaurant_name": "Demo Trattoria"}),
            ),
            AIMessage(content="This is a fictional restaurant example."),
        )
        result = restaurant_run(llm=llm)
        self.assertEqual(len(llm.bound_tools[0]), 4)
        self.assertEqual(len(result["tool_names_used"]), 4)
        self.assertEqual({t.name for t in RESTAURANT_TOOLS}, set(llm.bound_tools[0]))


if __name__ == "__main__":
    unittest.main()

