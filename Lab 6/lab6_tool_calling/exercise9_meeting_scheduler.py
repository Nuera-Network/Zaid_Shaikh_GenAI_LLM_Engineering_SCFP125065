"""Lab 6 Exercise 9: partial vs chained tool use, simulated booking only."""

from datetime import date as Date, timedelta

from langchain_core.tools import tool

from common import print_flow, run_tool_flow, used_tool_names


DEMO_BOOKINGS: set[tuple[str, str, str]] = set()
DEMO_SLOTS = ["10:00", "15:00"]
DEMO_OFFSETS = {"pune": "+05:30", "mumbai": "+05:30", "delhi": "+05:30",
                "goa": "+05:30", "kolkata": "+05:30"}


@tool
def check_calendar_availability(person: str, date: str) -> dict:
    """Check sample availability for a person on an ISO date (DEMO calendar only)."""
    parsed = Date.fromisoformat(date)
    if parsed < Date.today():
        raise ValueError("Cannot book dates in the past")
    available = [slot for slot in DEMO_SLOTS
                 if (person.casefold().strip(), date, slot) not in DEMO_BOOKINGS]
    return {"person": person, "date": date, "available": bool(available),
            "available_times": available, "source": "DEMO calendar; no real calendar consulted"}


@tool
def get_timezone_offset(city: str) -> dict:
    """Return a sample fixed UTC offset for an Indian city in this teaching demo."""
    offset = DEMO_OFFSETS.get(city.lower().strip())
    if offset is None:
        raise ValueError("Unsupported demo city; no time-zone assumption made")
    return {"city": city, "utc_offset": offset, "source": "DEMO mapping"}


@tool
def book_meeting(person: str, date: str, time: str) -> dict:
    """Book an available sample meeting slot in memory; no external calendar is changed."""
    Date.fromisoformat(date)
    person_key = person.lower().strip()
    key = (person_key, date, time)
    if time not in DEMO_SLOTS or key in DEMO_BOOKINGS:
        raise ValueError("Requested sample calendar slot is unavailable")
    DEMO_BOOKINGS.add(key)
    return {"booked": True, "person": person, "date": date, "time": time,
            "source": "IN-MEMORY DEMO ONLY; no real invitation sent"}


MEETING_TOOLS = [check_calendar_availability, get_timezone_offset, book_meeting]


def booking_guard(name: str, args: dict, prior: list[dict]) -> str | None:
    if name != "book_meeting":
        return None
    checked = [entry for entry in prior if entry["tool"] == "check_calendar_availability"
               and entry["success"] and entry["result"].get("person", "").casefold() ==
               str(args.get("person", "")).casefold() and
               entry["result"].get("date") == args.get("date")]
    tz_checked = any(entry["tool"] == "get_timezone_offset" and entry["success"]
                     for entry in prior)
    if not checked or not tz_checked:
        return "Check this person's availability and the city's time zone before booking."
    if args.get("time") not in checked[-1]["result"]["available_times"]:
        return "Selected slot was not reported as available."
    return None


def run(
    availability_query: str | None = None,
    booking_query: str | None = None,
    *,
    llm=None,
) -> dict:
    when = (Date.today() + timedelta(days=2)).isoformat()
    availability_query = availability_query or f"Is Zaid available on {when}? Just check, do not book."
    booking_query = booking_query or (
        f"Check Zaid's availability on {when}, check Pune's time-zone offset, "
        "then book 15:00 if available. This is a sample calendar."
    )
    first = run_tool_flow(
        availability_query, MEETING_TOOLS, llm=llm,
        system_extra="Only check availability when asked only to check. Never book without explicit request.",
        before_execute=booking_guard,
    )
    second = run_tool_flow(
        booking_query, MEETING_TOOLS, llm=llm,
        system_extra="For a booking request, check availability and timezone BEFORE booking. This is a DEMO calendar only.",
        before_execute=booking_guard,
    )
    return {
        "availability_only_flow": first,
        "booking_flow": second,
        "availability_tool_names": used_tool_names(first),
        "booking_tool_names": used_tool_names(second),
        "sample_bookings": sorted(DEMO_BOOKINGS),
    }


if __name__ == "__main__":
    print_flow(run())

