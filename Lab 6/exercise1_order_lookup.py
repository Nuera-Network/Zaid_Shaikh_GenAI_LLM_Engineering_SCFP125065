"""Lab 6 Exercise 1: @tool order lookup and manual tool-call flow."""

from langchain_core.tools import tool

from common import print_flow, run_tool_flow


DEMO_ORDERS = {
    "ORD-4521": "In transit; estimated delivery in 2 days",
    "ORD-1200": "Delivered",
    "ORD-9001": "Processing at warehouse",
}


@tool
def lookup_order_status(order_id: str) -> dict:
    """Look up a sample order's shipping status by order ID, e.g. ORD-4521."""
    status = DEMO_ORDERS.get(order_id.upper().strip())
    return {
        "order_id": order_id.upper().strip(),
        "found": status is not None,
        "status": status or "Unknown order ID",
        "data_type": "DEMO SAMPLE DATA, not a real courier lookup",
    }


def run(user_query: str = "Where is my order ORD-4521?", *, llm=None) -> dict:
    return run_tool_flow(user_query, [lookup_order_status], llm=llm)


if __name__ == "__main__":
    print_flow(run())

