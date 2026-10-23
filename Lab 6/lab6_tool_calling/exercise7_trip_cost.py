"""Lab 6 Exercise 7: three independent travel-cost tools for one query."""

from langchain_core.tools import tool

from common import print_flow, run_tool_flow, used_tool_names


# Explicitly fictional teaching rates. Never represented as live market prices.
DEMO_ROUND_TRIP_FLIGHTS = {("delhi", "goa"): 8200, ("mumbai", "goa"): 5800}
DEMO_HOTEL_PER_NIGHT = {"goa": 3200, "pune": 2700, "mumbai": 4200}
DEMO_FOOD_PER_DAY = {"goa": 1100, "pune": 850, "mumbai": 1200}


@tool
def get_flight_cost(origin: str, destination: str) -> dict:
    """Look up an illustrative round-trip flight budget for one person in INR."""
    route = (origin.strip().lower(), destination.strip().lower())
    amount = DEMO_ROUND_TRIP_FLIGHTS.get(route)
    return {"origin": origin, "destination": destination, "cost_inr": amount,
            "source": "DEMO ESTIMATE, not a flight quote", "available": amount is not None}


@tool
def get_hotel_cost(city: str, nights: int) -> dict:
    """Estimate a demo hotel stay budget in INR for a city and number of nights."""
    if nights <= 0:
        raise ValueError("nights must be positive")
    per_night = DEMO_HOTEL_PER_NIGHT.get(city.strip().lower())
    return {"city": city, "nights": nights, "cost_inr": per_night * nights if per_night else None,
            "source": "DEMO ESTIMATE, not a hotel booking", "available": per_night is not None}


@tool
def get_food_budget(city: str, days: int) -> dict:
    """Estimate a demo food budget in INR for a city and number of days."""
    if days <= 0:
        raise ValueError("days must be positive")
    per_day = DEMO_FOOD_PER_DAY.get(city.strip().lower())
    return {"city": city, "days": days, "cost_inr": per_day * days if per_day else None,
            "source": "DEMO ESTIMATE, not a restaurant quote", "available": per_day is not None}


TRIP_TOOLS = [get_flight_cost, get_hotel_cost, get_food_budget]


def run(
    user_query: str = "I'm planning a 3-night trip from Delhi to Goa. What will flights, hotel, and food roughly cost?",
    *,
    llm=None,
) -> dict:
    flow = run_tool_flow(
        user_query, TRIP_TOOLS, llm=llm,
        system_extra="Request all three independent budget tools. These are illustrative rates only, not live travel quotes.",
    )
    used = used_tool_names(flow)
    expected = {tool.name for tool in TRIP_TOOLS}
    valid_costs = [
        item["result"]["cost_inr"] for round_ in flow["rounds"]
        for item in round_["manual_results"]
        if item["tool"] in expected and item["success"]
        and isinstance(item["result"], dict) and item["result"].get("cost_inr") is not None
    ]
    flow["tools_used"] = used
    flow["all_three_called"] = expected.issubset(set(used))
    flow["python_calculated_total_inr"] = sum(valid_costs) if len(valid_costs) == 3 else None
    return flow


if __name__ == "__main__":
    print_flow(run())

