# Lab 6 — Tools and Tool Calling with LLMs

Ten complete exercises using **LangChain tools and ChatGroq**. They demonstrate
how an LLM *proposes* tool calls and a Python application *executes* them,
passes back tool outputs as `ToolMessage` objects, and obtains a final answer.

## Assignment checklist

| # | Exercise | Required tool construction | Demonstration |
| --- | --- | --- | --- |
| 1 | Order Status | `@tool` | Order ID → call → sample status → answer |
| 2 | Loan EMI | `StructuredTool` + Pydantic `args_schema` | Validates principal/rate/tenure and calculates EMI |
| 3 | Unit Converter | Two `@tool` functions | Model chooses temperature or distance tool |
| 4 | Weather Outfit | Two `@tool` functions | Chained live weather → outfit; compares model arguments with retrieved weather |
| 5 | Async Stock | `BaseTool` subclass | `_run`, `_arun`, `.invoke()`, `.ainvoke()`, plus bound-model flow |
| 6 | Expense Split | `StructuredTool` + Pydantic `args_schema` | Direct call versus model-chosen call, compares results |
| 7 | Trip Cost | Three `@tool` functions | Flight + hotel + food in one query; Python also checks combined total |
| 8 | Health Metrics | Two `StructuredTool` instances in `HealthToolkit` | BMI and estimated daily calorie range |
| 9 | Meeting Scheduler | Three `@tool` functions | Availability-only versus checked time-zone and **simulated** booking |
| 10 | Restaurant Assistant | Four `@tool` functions | Search → ratings, hours and price for fictional restaurants |

Each exercise file's `run()` returns the proposed `tool_calls`, Python's
`manual_results`, and the model's `final_answer` per conversation round.
`common.py` implements the manual `bind_tools()` loop. It keeps the original
`tool_call_id` and supports additional validation before tool execution.

## Setup in VS Code (Windows)

Tested using **Python 3.12**. If dependencies fail under Python 3.14, select
Python 3.12 when creating the environment.

1. Extract the ZIP and open the `lab6_tool_calling` folder in VS Code.
2. Install dependencies in a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`; insert your own Groq API key after
   `GROQ_API_KEY=`. The `.gitignore` prevents accidentally committing `.env`.
4. Execute an individual script (recommended to avoid rate limits):

   ```powershell
   .venv\Scripts\python.exe -X utf8 exercise1_order_lookup.py
   .venv\Scripts\python.exe -X utf8 exercise4_weather_outfit.py
   ```

5. Or run all ten exercises one after another:

   ```powershell
   .venv\Scripts\python.exe -X utf8 run_all_examples.py
   ```

6. Run automated tests **without** any Groq API key or weather network call:

   ```powershell
   .venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

The `-X utf8` switch avoids the Windows `cp1252` Unicode issue encountered
earlier when printing Marathi/Hindi symbols or the rupee sign.

## Important concepts demonstrated

### Manual model-to-tool handshake

1. Register tools with `bound = llm.bind_tools(tools)`.
2. Send a user's question to `bound.invoke(messages)`.
3. Inspect the model's proposed `AIMessage.tool_calls` with names/arguments.
4. Execute each approved tool manually in Python via `tool.invoke(args)`.
5. Append one `ToolMessage` containing the result and its original ID.
6. Ask the model again. Repeat as needed, then print the final answer.

This distinguishes *LLM tool selection* from *Python tool execution*. The LLM
does not secretly run code or call external systems by itself.

### Input validation

Loan EMI uses principal greater than zero, annual rate in `[0,30]`, and tenure
greater than zero. Expense splitting requires positive bill and people count.
Bad inputs raise validation errors, which the tool loop reports as failures.

The EMI assumes a fixed nominal annual interest rate with monthly repayment;
fees and insurance are excluded. It is a calculation demo, not a loan quote.
Equal split rounding can leave a one-paise remainder to allocate separately.

### Dependent versus independent tools

- **Exercise 4:** `recommend_outfit` is blocked until weather has been fetched.
  The validator rejects invented temperatures or conditions and asks the model
  to reuse exact retrieved values.
- **Exercise 7:** Three independent cost tools may be proposed in one round;
  all three results go back before the final combined answer. Rates are sample
  estimates only; no flight/hotel inventory is queried.
- **Exercise 9:** Booking is guarded by prior availability and timezone checks.
  It only modifies an in-memory Python set and **never creates a real event or
  sends invitations**.
- **Exercise 10:** Names, ratings, prices and opening days are invented example
  data. Do not mistake them for current Pune restaurants.

### Current weather

Exercise 4 retrieves model-based **current** conditions from
[Open-Meteo](https://open-meteo.com/en/docs) after geocoding the city. It needs
internet access. Forecast/model conditions may differ from direct thermometer
observations, and an ambiguous city name may select the wrong place.

### Health metrics

BMI is a rough screening statistic, not a medical diagnosis. The Mifflin-St
Jeor adult daily-calorie calculation normally depends on sex, which the lab
input omits. To avoid inventing sex, the tool returns a **range covering the
two standard equation constants**, scaled by activity level. This is not an
individual dietary recommendation.

## Testing and submission

The bundled tests simulate model messages, ensuring tools receive the right
arguments and their outputs are returned to the model. They also test EMI,
unit conversions, async stock checking, tool ordering, demo booking, and more.
The tests **do not prove that a live model will always select the intended
tools**. Run each live script with your own API key and inspect output.

Copy the `lab6_tool_calling` folder into your existing
`generative_ai_<student-name-or-roll-no>` GitHub repository. Commit exercise
files separately if your instructor requires one push per exercise. Never
include `.env` in a commit.

```bash
git add lab6_tool_calling
git commit -m "Complete Lab 6 tools and tool calling exercises"
git push
```

### Official documentation

- [LangChain Python tools reference](https://reference.langchain.com/python/langchain-core/tools)
- [LangChain ChatGroq integration](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [Open-Meteo current weather](https://open-meteo.com/en/docs)

