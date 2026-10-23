"""Lab 6 Exercise 5: synchronous + asynchronous BaseTool stock checker."""

import asyncio

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from common import print_flow, run_tool_flow


DEMO_STOCK = {"P1023": 14, "P1000": 0, "P2000": 7}


class StockArguments(BaseModel):
    product_id: str = Field(description="Product SKU such as P1023")


class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = "Check whether a product ID is in stock in the sample store inventory."
    args_schema: type[BaseModel] = StockArguments

    def _run(self, product_id: str, run_manager=None) -> dict:
        quantity = DEMO_STOCK.get(product_id.upper().strip())
        return {
            "product_id": product_id.upper().strip(),
            "found": quantity is not None,
            "in_stock": quantity is not None and quantity > 0,
            "quantity": quantity,
            "data_type": "DEMO inventory, not a real store",
        }

    async def _arun(self, product_id: str, run_manager=None) -> dict:
        await asyncio.sleep(0)  # Async interface; real apps await database I/O.
        return self._run(product_id)


stock_tool = ProductStockTool()


async def run(
    product_id: str = "P1023",
    user_query: str = "Is product P1023 in stock?",
    *,
    llm=None,
) -> dict:
    sync_result = stock_tool.invoke({"product_id": product_id})
    async_result = await stock_tool.ainvoke({"product_id": product_id})
    flow = run_tool_flow(user_query, [stock_tool], llm=llm)
    return {
        "manual_sync_invoke": sync_result,
        "manual_async_ainvoke": async_result,
        "manual_results_equal": sync_result == async_result,
        "bound_model_flow": flow,
    }


if __name__ == "__main__":
    print_flow(asyncio.run(run()))

