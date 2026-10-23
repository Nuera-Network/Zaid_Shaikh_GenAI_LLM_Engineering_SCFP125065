"""Run all ten Lab 6 demonstrations with one configured Groq model."""

import asyncio
import json

from common import get_llm
from exercise1_order_lookup import run as exercise1
from exercise2_loan_emi import run as exercise2
from exercise3_unit_converter import run as exercise3
from exercise4_weather_outfit import run as exercise4
from exercise5_async_stock import run as exercise5
from exercise6_expense_split import run as exercise6
from exercise7_trip_cost import run as exercise7
from exercise8_health_toolkit import run as exercise8
from exercise9_meeting_scheduler import run as exercise9
from exercise10_restaurant_assistant import run as exercise10


async def main():
    llm = get_llm()
    runs = [
        ("Order Status", lambda: exercise1(llm=llm)),
        ("Loan EMI", lambda: exercise2(llm=llm)),
        ("Unit Converter", lambda: exercise3(llm=llm)),
        ("Weather Outfit", lambda: exercise4(llm=llm)),
        ("Async Stock", lambda: exercise5(llm=llm)),
        ("Expense Splitter", lambda: exercise6(llm=llm)),
        ("Trip Cost", lambda: exercise7(llm=llm)),
        ("Health Toolkit", lambda: exercise8(llm=llm)),
        ("Meeting Scheduler", lambda: exercise9(llm=llm)),
        ("Restaurant Assistant", lambda: exercise10(llm=llm)),
    ]
    for index, (name, runner) in enumerate(runs, 1):
        print(f"\n{'=' * 12} EXERCISE {index}: {name} {'=' * 12}")
        try:
            result = await runner() if index == 5 else runner()
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        except Exception as exc:
            print(f"Exercise failed: {type(exc).__name__}: {exc}")
            print("Check API availability, model responses and input assumptions.")


if __name__ == "__main__":
    asyncio.run(main())

