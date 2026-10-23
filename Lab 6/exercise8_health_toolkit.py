"""Lab 6 Exercise 8: BMI + indicative calorie range via HealthToolkit."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common import print_flow, run_tool_flow, used_tool_names


class BMIArguments(BaseModel):
    weight_kg: float = Field(gt=0, le=400, allow_inf_nan=False)
    height_cm: float = Field(gt=0, le=260, allow_inf_nan=False)


class CalorieArguments(BMIArguments):
    age: int = Field(gt=0, le=120)
    activity_level: str = Field(description="One of sedentary, light, moderate, active, very_active")


ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55,
                    "active": 1.725, "very_active": 1.9}


def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate the numeric body mass index from height and weight."""
    BMIArguments(weight_kg=weight_kg, height_cm=height_cm)
    bmi = weight_kg / (height_cm / 100) ** 2
    return {"bmi": round(bmi, 2), "units": "kg/m²",
            "note": "BMI is a limited screening measure; not a diagnosis"}


def calculate_daily_calories(weight_kg: float, height_cm: float, age: int, activity_level: str) -> dict:
    """Estimate adult maintenance calories as a range without assuming sex."""
    CalorieArguments(weight_kg=weight_kg, height_cm=height_cm, age=age,
                     activity_level=activity_level)
    key = activity_level.lower().strip().replace(" ", "_")
    if key not in ACTIVITY_FACTORS:
        raise ValueError(f"activity_level must be one of: {', '.join(ACTIVITY_FACTORS)}")
    if age < 18:
        raise ValueError("This classroom calculation is for adults aged 18+ only")
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    factor = ACTIVITY_FACTORS[key]
    return {
        "estimated_daily_maintenance_kcal_range": [round((base - 161) * factor),
                                                  round((base + 5) * factor)],
        "activity_level": key,
        "method": "Mifflin-St Jeor female/male equation bounds × activity multiplier",
        "note": "Illustrative adult estimate, not a personal medical or diet recommendation",
    }


class HealthToolkit:
    """Groups the two StructuredTool instances required by the exercise."""

    def __init__(self):
        self.tools = [
            StructuredTool.from_function(
                func=calculate_bmi, name="calculate_bmi",
                description="Compute numeric BMI using kg and centimeters.",
                args_schema=BMIArguments,
            ),
            StructuredTool.from_function(
                func=calculate_daily_calories, name="calculate_daily_calories",
                description="Estimate adult daily calorie maintenance range using weight, height, age and activity level.",
                args_schema=CalorieArguments,
            ),
        ]

    def get_tools(self):
        return self.tools


def run(
    user_query: str = "I'm 70kg, 175cm, 25 years old and moderately active. What are my BMI and daily calorie needs?",
    *,
    llm=None,
) -> dict:
    tools = HealthToolkit().get_tools()
    flow = run_tool_flow(user_query, tools, llm=llm)
    flow["toolkit_tool_names"] = [tool.name for tool in tools]
    flow["both_tools_called"] = set(flow["toolkit_tool_names"]).issubset(set(used_tool_names(flow)))
    return flow


if __name__ == "__main__":
    print_flow(run())

