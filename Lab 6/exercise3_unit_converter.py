"""Lab 6 Exercise 3: two @tool converters bound together."""

from typing import Literal

from langchain_core.tools import tool

from common import print_flow, run_tool_flow, used_tool_names


@tool
def convert_temperature(value: float, from_unit: Literal["C", "F"], to_unit: Literal["C", "F"]) -> dict:
    """Convert a temperature between Celsius (C) and Fahrenheit (F)."""
    if from_unit == to_unit:
        result = value
    elif from_unit == "C":
        result = value * 9 / 5 + 32
    else:
        result = (value - 32) * 5 / 9
    return {"input": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 4)}


@tool
def convert_distance(value: float, from_unit: Literal["km", "miles"], to_unit: Literal["km", "miles"]) -> dict:
    """Convert a distance between kilometers (km) and miles."""
    if from_unit == to_unit:
        result = value
    elif from_unit == "km":
        result = value / 1.609344
    else:
        result = value * 1.609344
    return {"input": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 4)}


def run(user_query: str = "Convert 100 km to miles", *, llm=None) -> dict:
    flow = run_tool_flow(user_query, [convert_temperature, convert_distance], llm=llm)
    flow["selected_tools"] = used_tool_names(flow)
    return flow


if __name__ == "__main__":
    print_flow(run())

