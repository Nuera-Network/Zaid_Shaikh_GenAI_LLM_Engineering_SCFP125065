"""Shared prompts, parsers and Groq setup for Lab 5 workflows."""

from __future__ import annotations

import os
from typing import Type

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel


def get_llm():
    """Initialize the Groq model after verifying the environment is configured."""
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY. Copy .env.example to .env first.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=3000,
    )


def extraction_chain(llm, schema: Type[BaseModel], instructions: str, variables: list[str]):
    """PromptTemplate -> LLM -> PydanticOutputParser (raw-data extraction)."""
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = PromptTemplate(
        template=instructions + "\n\n{format_instructions}",
        input_variables=variables,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def writing_chain(llm, persona: str, task_prompt: str):
    """ChatPromptTemplate -> LLM -> StrOutputParser (free-form text)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona),
        ("human", task_prompt),
    ])
    return prompt | llm | StrOutputParser()


def evaluation_chain(llm, schema: Type[BaseModel], persona: str, task_prompt: str):
    """ChatPromptTemplate -> LLM -> PydanticOutputParser (structured review)."""
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona),
        ("human", task_prompt + "\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def source_json(value: BaseModel) -> str:
    """Pass only validated structured data between steps, not raw source text."""
    return value.model_dump_json(indent=2)

