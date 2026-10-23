"""Pydantic contracts required by every Lab 5 step."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComplaintData(BaseModel):
    complaint: str
    product_or_feature: str
    sentiment: Literal["positive", "negative", "neutral", "mixed"]


class ResearchData(BaseModel):
    research_question: str
    method: str
    key_finding: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class InterviewAnswers(BaseModel):
    answers: list[QuestionAnswer]


class SkillEvidence(BaseModel):
    skill: str
    demonstrated: bool
    evidence_or_gap: str


class SkillEvaluation(BaseModel):
    assessments: list[SkillEvidence]


class HiringScorecard(BaseModel):
    rows: list[SkillEvidence]
    overall_recommendation: str


class EventDetails(BaseModel):
    event_type: str
    event_date: str = Field(description="ISO date in YYYY-MM-DD format")
    location: str
    indoor_outdoor_preference: Literal["indoor", "outdoor", "flexible"]


class WeatherData(BaseModel):
    temperature: float = Field(description="Daily maximum temperature in Celsius")
    precipitation_chance: float = Field(ge=0, le=100, description="Daily maximum probability, percent")
    conditions: str


class SupportSummary(BaseModel):
    issue: str
    customer_tone: str
    resolution_attempted: str


class EscalationGaps(BaseModel):
    missing_fields: list[str]
    is_ready_to_escalate: bool


class FitnessGoal(BaseModel):
    goal_type: str
    fitness_level: str
    days_available: int = Field(ge=1, le=7)
    equipment_access: str


class WorkoutDay(BaseModel):
    day_number: int = Field(ge=1, le=7)
    activity: str
    duration_minutes: int = Field(ge=5, le=180)
    intensity: Literal["low", "moderate", "high"]


class WorkoutPlan(BaseModel):
    days: list[WorkoutDay]

