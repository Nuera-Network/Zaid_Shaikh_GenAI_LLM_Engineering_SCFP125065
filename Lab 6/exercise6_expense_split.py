"""Lab 6 Exercise 6: StructuredTool expense split vs direct invocation."""

from decimal import Decimal, ROUND_HALF_UP

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common import print_flow, run_tool_flow


class SplitArguments(BaseModel):
    total_amount: float = Field(gt=0, allow_inf_nan=False, description="Total bill in INR")
    num_people: int = Field(gt=0, description="Number of people splitting the bill")


def split_expense(total_amount: float, num_people: int) -> dict:
    """Split an INR bill equally; show a rounding remainder if division is uneven."""
    SplitArguments(total_amount=total_amount, num_people=num_people)
    bill = Decimal(str(total_amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    share = (bill / Decimal(num_people)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    remainder = bill - share * num_people
    return {
        "total_amount_inr": str(bill),
        "num_people": num_people,
        "per_person_inr": str(share),
        "rounding_remainder_inr": str(remainder),
        "note": "Remainder may need allocating to one person for exact settlement",
    }


expense_tool = StructuredTool.from_function(
    func=split_expense,
    name="split_expense",
    description="Split a total INR bill equally between a positive number of people.",
    args_schema=SplitArguments,
)


def run(
    user_query: str = "Split ₹2400 between 4 friends",
    total_amount: float = 2400,
    num_people: int = 4,
    *,
    llm=None,
) -> dict:
    manual = expense_tool.invoke({"total_amount": total_amount, "num_people": num_people})
    flow = run_tool_flow(user_query, [expense_tool], llm=llm)
    bound_results = [
        item["result"] for round_ in flow["rounds"]
        for item in round_["manual_results"]
        if item["tool"] == "split_expense" and item["success"]
    ]
    return {
        "manual_result": manual,
        "bound_model_flow": flow,
        "manual_matches_bound_result": bool(bound_results) and manual == bound_results[0],
    }


if __name__ == "__main__":
    print_flow(run())

