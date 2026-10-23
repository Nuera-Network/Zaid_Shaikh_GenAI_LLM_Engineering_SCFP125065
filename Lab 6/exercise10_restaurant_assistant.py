"""Lab 6 Exercise 10: four restaurant tools bound in one call."""

from datetime import date

from langchain_core.tools import tool

from common import print_flow, run_tool_flow, used_tool_names


# Fictional venues and illustrative data prevent fake real-world claims.
DEMO_RESTAURANTS = {
    "Demo Trattoria": {"cuisine": "italian", "city": "pune", "rating": 4.4,
                       "open_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"],
                       "cost_for_two_inr": 1500},
    "Sample Pasta House": {"cuisine": "italian", "city": "pune", "rating": 4.1,
                           "open_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                           "cost_for_two_inr": 1100},
    "Example Spice Cafe": {"cuisine": "indian", "city": "pune", "rating": 4.2,
                           "open_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                           "cost_for_two_inr": 850},
}


@tool
def find_restaurants(cuisine: str, city: str) -> dict:
    """Find fictional teaching-example restaurants by cuisine and city."""
    names = [name for name, data in DEMO_RESTAURANTS.items()
             if data["cuisine"] == cuisine.lower().strip() and data["city"] == city.lower().strip()]
    return {"restaurants": names, "source": "FICTIONAL DEMO VENUES"}


@tool
def get_restaurant_rating(restaurant_name: str) -> dict:
    """Get a fictional restaurant's sample rating out of 5."""
    details = DEMO_RESTAURANTS.get(restaurant_name)
    return {"restaurant": restaurant_name, "rating_out_of_5": details["rating"] if details else None,
            "source": "FICTIONAL DEMO RATINGS"}


@tool
def check_restaurant_open(restaurant_name: str, day: str) -> dict:
    """Check a fictional restaurant's example weekly schedule for a weekday name."""
    details = DEMO_RESTAURANTS.get(restaurant_name)
    return {"restaurant": restaurant_name, "day": day,
            "open": day.lower().strip() in details["open_days"] if details else None,
            "source": "FICTIONAL DEMO SCHEDULE; not live opening hours"}


@tool
def estimate_cost_for_two(restaurant_name: str) -> dict:
    """Get a fictional restaurant's estimated INR price for two diners."""
    details = DEMO_RESTAURANTS.get(restaurant_name)
    return {"restaurant": restaurant_name,
            "cost_for_two_inr": details["cost_for_two_inr"] if details else None,
            "source": "FICTIONAL DEMO PRICE"}


RESTAURANT_TOOLS = [find_restaurants, get_restaurant_rating,
                    check_restaurant_open, estimate_cost_for_two]


def run(user_query: str | None = None, *, llm=None) -> dict:
    day = date.today().strftime("%A")
    query = user_query or (
        f"Suggest an Italian restaurant in Pune open on {day}, with its rating "
        "and rough cost for two. Use the bound example restaurant data."
    )
    flow = run_tool_flow(
        query, RESTAURANT_TOOLS, llm=llm,
        system_extra="All restaurant names, ratings, hours and prices here are FICTIONAL classroom data, not actual businesses.",
    )
    flow["bound_tool_names"] = [tool.name for tool in RESTAURANT_TOOLS]
    flow["tool_names_used"] = used_tool_names(flow)
    return flow


if __name__ == "__main__":
    print_flow(run())

