# Lab 5 — Workflow Design with LLMs

Six Groq/LangChain pipelines implementing the exact **PromptTemplate**,
**ChatPromptTemplate**, **PydanticOutputParser**, and **StrOutputParser**
combinations specified in the provided Lab 5 brief.

Each stage consumes only the preceding stage's validated output, not the
original raw input (except Stage 1). Exercise 4 obtains weather from a real API.

## Assignment compliance

| Exercise | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| 1. Review → Ticket | PromptTemplate + PydanticOutputParser | ChatPromptTemplate (support ticket writer) + StrOutputParser | — |
| 2. Abstract → Layperson | PromptTemplate + PydanticOutputParser | ChatPromptTemplate (science communicator) + StrOutputParser | — |
| 3. Interview → Scorecard | PromptTemplate + PydanticOutputParser | ChatPromptTemplate (technical interviewer) + PydanticOutputParser | ChatPromptTemplate (hiring coordinator) + PydanticOutputParser |
| 4. Event → Weather Recommendation | PromptTemplate + PydanticOutputParser | PromptTemplate → Open-Meteo API → PydanticOutputParser | ChatPromptTemplate (event planner) + StrOutputParser |
| 5. Chat → Escalation | PromptTemplate + PydanticOutputParser | ChatPromptTemplate (escalation reviewer) + PydanticOutputParser | ChatPromptTemplate (support operations lead) + StrOutputParser |
| 6. Goal → Workout Plan | PromptTemplate + PydanticOutputParser | ChatPromptTemplate (personal trainer) + PydanticOutputParser | — |

## Files

- `common.py`: reusable LangChain prompt and output-parser builders; Groq setup.
- `schemas.py`: typed Pydantic contracts for every structured step.
- `weather_api.py`: real forecast retrieval through Open-Meteo (geocoding + daily forecast).
- `exercise1_review_to_ticket.py` through `exercise6_weekly_workout.py`: one solution per exercise.
- `run_all_examples.py`: demonstrations on invented data.
- `tests/test_lab5.py`: offline mock-model tests; no API key needed.

## Setup (Windows VS Code / Python)

1. Open the `lab5_workflow_design` folder in VS Code.
   The project was tested with Python 3.12; if a package does not install on
   Python 3.14, select a Python 3.12 interpreter for the virtual environment.
2. Run in the integrated terminal:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   If Windows blocks activation, use `.venv\Scripts\python.exe` instead of
   `python` in commands below; no execution-policy change is required.

3. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. Do not commit
   the key or include it in screenshots.
4. Run all examples:

   ```powershell
   python -X utf8 run_all_examples.py
   ```

   The `-X utf8` flag prevents the `cp1252` console encoding error you saw
   earlier when printing Hindi text.

5. Run an individual exercise:

   ```powershell
   python -X utf8 exercise3_interview_scorecard.py
   ```

6. Run offline verification:

   ```powershell
   python -m unittest discover -s tests -v
   ```

## How the chains pass data

**Exercise 1**

`review_text → complaint: ComplaintData → support ticket: str`

**Exercise 2**

`abstract_text → ResearchData → layperson summary: str`

**Exercise 3**

`transcript_text → InterviewAnswers → SkillEvaluation → HiringScorecard`

**Exercise 4**

`event_description_text → EventDetails → formatted API query → Open-Meteo → WeatherData → recommendation: str`

**Exercise 5**

`chat_log_text → SupportSummary → EscalationGaps → report: str`

**Exercise 6**

`fitness_goal_text + days_per_week → FitnessGoal → WorkoutPlan`

## Reliability notes

- Outputs are parsed into Pydantic objects. If a model returns malformed JSON,
  parsing raises an error; it never silently treats malformed text as trusted.
- Hiring evidence is limited to documented interview answers, with missing
  skills marked as gaps. The recommendation is for human review only.
- Workout plans must have exactly the provided number of available days.
- In Exercise 4, the LLM never invents weather: the API response is normalized
  in Python and parsed by `PydanticOutputParser`. Dates beyond the forecast
  horizon, unknown locations, API failures or missing readings cause an
  explicit exception. The live example catches weather failures and proceeds
  with the remaining exercises; the standalone script reports the error.
- The geocoding API uses the top matching place; add country/region in the
  description to reduce location ambiguity. Verify the returned location for
  critical outdoor events.
- Open-Meteo provides daily maximum temperature in °C and maximum daily
  precipitation probability in %. Weather is a forecast, not a guarantee.
- All provided example content is fictional. Don't send confidential resumes,
  chats, or transcripts to a hosted model without authorization.

## Live execution versus offline tests

`tests/test_lab5.py` uses a simulated LangChain model and a simulated weather
provider. This verifies parser usage and stage-to-stage data handoffs without
network access. It **does not** establish that the live Groq model will always
produce correct answers or that the live weather service is available.

Once your API key is configured, run the example scripts, inspect every stage,
and retain screenshots as execution evidence if required by your instructor.

## GitHub submission

Copy this whole folder into your **existing**
`generative_ai_<student-name-or-roll-no>` repository. Use a separate commit
per exercise if your instructor requires pushing work incrementally, and push
the shared helpers, tests and README as well. Never upload the `.env` file.

```bash
git add lab5_workflow_design
git commit -m "Complete Lab 5 workflow design with LLMs"
git push
```

Final link: `https://github.com/<username>/generative_ai_<student-name-or-roll-no>`

## Official references

- [LangChain: Groq Chat Model](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo geocoding API](https://open-meteo.com/en/docs/geocoding-api)

