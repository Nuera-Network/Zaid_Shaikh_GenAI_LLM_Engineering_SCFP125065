"""Lab 6 Exercise 2: validated loan EMI StructuredTool."""

from math import isfinite

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common import print_flow, run_tool_flow


class EMIArguments(BaseModel):
    principal: float = Field(gt=0, allow_inf_nan=False, description="Loan principal in INR")
    annual_rate: float = Field(ge=0, le=30, allow_inf_nan=False, description="Annual interest percent")
    tenure_months: int = Field(gt=0, description="Number of monthly payments")


def calculate_loan_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """Calculate fixed-rate monthly loan EMI (does not include fees or insurance)."""
    EMIArguments(principal=principal, annual_rate=annual_rate, tenure_months=tenure_months)
    rate = annual_rate / 1200
    if rate == 0:
        payment = principal / tenure_months
    else:
        # Algebraically equivalent to P*r*(1+r)^n / ((1+r)^n - 1).
        discount = (1 + rate) ** (-tenure_months)
        payment = principal * rate / (1 - discount)
    if not isfinite(payment):
        raise ValueError("EMI cannot be represented for these values")
    return {
        "monthly_emi_inr": round(payment, 2),
        "total_payment_inr": round(payment * tenure_months, 2),
        "total_interest_inr": round(payment * tenure_months - principal, 2),
        "assumptions": "Fixed annual nominal rate; monthly repayments; excludes fees",
    }


emi_tool = StructuredTool.from_function(
    func=calculate_loan_emi,
    name="calculate_loan_emi",
    description="Calculate EMI for a loan using principal INR, annual interest rate percent, and tenure months.",
    args_schema=EMIArguments,
)


def run(user_query: str = "What's my EMI for a 5 lakh loan at 9% for 24 months?", *, llm=None) -> dict:
    return run_tool_flow(user_query, [emi_tool], llm=llm)


if __name__ == "__main__":
    print_flow(run())

