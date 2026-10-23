"""Lab 6 Exercise 4: two dependent tools with argument provenance checks."""

from langchain_core.tools import tool

from common import print_flow, run_tool_flow
from weather_live import current_weather


@tool
def get_weather(city: str) -> dict:
    """Retrieve model-based current temperature Celsius and conditions for a city via Open-Meteo."""
    return current_weather(city)


@tool
def recommend_outfit(temperature: float, conditions: str) -> dict:
    """Recommend an outfit from the measured Celsius temperature and observed conditions."""
    clothing = []
    if temperature < 10:
        clothing.append("warm coat and layered clothes")
    elif temperature < 20:
        clothing.append("light jacket or sweater")
    elif temperature < 30:
        clothing.append("comfortable breathable clothes")
    else:
        clothing.append("lightweight, breathable clothes")
    lowered = conditions.lower()
    if "rain" in lowered or "storm" in lowered:
        clothing.append("umbrella or waterproof layer")
    if "snow" in lowered:
        clothing.append("insulated footwear")
    return {"temperature": temperature, "conditions": conditions, "outfit": ", ".join(clothing)}


def ensure_weather_provenance(name: str, args: dict, executed: list[dict]) -> str | None:
    if name != "recommend_outfit":
        return None
    weather_results = [e["result"] for e in executed if e["tool"] == "get_weather" and e["success"]]
    if not weather_results:
        return "First call get_weather and use its actual output before recommending an outfit."
    weather = weather_results[-1]
    if (
        abs(float(args.get("temperature", -9999)) - weather["temperature"]) > 0.01
        or str(args.get("conditions", "")).casefold() != weather["conditions"].casefold()
    ):
        return "Outfit arguments did not match fetched weather. Reuse exact temperature and conditions."
    return None


def run(user_query: str = "What should I wear today in Mumbai?", *, llm=None) -> dict:
    flow = run_tool_flow(
        user_query, [get_weather, recommend_outfit], llm=llm,
        system_extra="Call get_weather first. Then use its exact temperature and conditions to call recommend_outfit.",
        before_execute=ensure_weather_provenance,
    )
    flow["outfit_used_verified_weather"] = any(
        item["tool"] == "recommend_outfit" and item["success"]
        for round_ in flow["rounds"] for item in round_["manual_results"]
    )
    return flow


if __name__ == "__main__":
    print_flow(run())

