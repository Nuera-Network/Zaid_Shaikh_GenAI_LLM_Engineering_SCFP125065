"""Execute every Lab 5 pipeline on synthetic example data using Groq."""

import json
from datetime import date, timedelta

from common import get_llm
from exercise1_review_to_ticket import review_to_ticket
from exercise2_abstract_to_summary import abstract_to_summary
from exercise3_interview_scorecard import interview_to_scorecard
from exercise4_event_weather import event_weather_recommendation
from exercise5_support_escalation import support_escalation
from exercise6_weekly_workout import workout_goal_to_plan


def display(title: str, result: dict):
    print("\n" + "=" * 12 + " " + title + " " + "=" * 12)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    llm = get_llm()
    display("EXERCISE 1", review_to_ticket(
        "The headphone audio is excellent but the Bluetooth keeps disconnecting.", llm=llm
    ))
    display("EXERCISE 2", abstract_to_summary(
        "Researchers investigated whether short study breaks improve recall. "
        "A randomized study with 60 students compared breaks against no breaks. "
        "The breaks group recalled more words on an immediate quiz.", llm=llm
    ))
    display("EXERCISE 3", interview_to_scorecard(
        "Interviewer: How do you test Python? Candidate: I write unittest cases. "
        "Interviewer: Can you explain SQL JOIN? Candidate: Not yet.",
        ["Python testing", "SQL JOIN"], llm=llm
    ))

    # Live weather is requested separately. A future forecast can only be
    # retrieved for dates within the provider's supported horizon.
    when = (date.today() + timedelta(days=3)).isoformat()
    try:
        display("EXERCISE 4", event_weather_recommendation(
            f"We're planning an outdoor college music performance in Pune, India on {when}.",
            llm=llm,
        ))
    except Exception as exc:
        print(f"\nEXERCISE 4 weather unavailable: {type(exc).__name__}: {exc}")

    display("EXERCISE 5", support_escalation(
        "Customer: Checkout fails every time. Agent: Have you restarted? "
        "Customer: Yes, still broken. Agent: I will escalate it.", llm=llm
    ))
    display("EXERCISE 6", workout_goal_to_plan(
        "I'm a beginner who wants to build stamina. No gym equipment at home.",
        days_per_week=3, llm=llm
    ))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nLab run stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

