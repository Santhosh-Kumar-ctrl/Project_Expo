import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from graph_schema import QueryClassification
from config import CLASSIFICATION_CONFIDENCE_THRESHOLD

load_dotenv()

_classifier_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

CLASSIFICATION_PROMPT = """You are a query router for a policy document retrieval system.

Classify the user's query into exactly one category:

- LOW_CHUNKS: Simple factual lookups, definitions, single-topic questions that need 1-2 document excerpts.
  Examples: "What is the attendance policy?", "Define academic probation.", "What is the minimum GPA?"

- HIGH_CHUNKS: Complex questions requiring synthesis across multiple sections, comparisons, or detailed explanations.
  Examples: "Compare the undergraduate and graduate grading scales", "Explain all eligibility requirements for examinations.", "List all consequences of academic probation."

- GRAPH_RETRIEVAL: Questions about impacts, dependencies, consequences of changes, what-if scenarios, or cascading effects of modifying a policy/requirement/threshold.
  Examples: "What happens if we change attendance from 70% to 75%?", "Which policies are affected if the credit hour requirement changes?", "What is the impact of removing the medical leave exception?"

Classify the following query."""


def classify_query(query: str) -> QueryClassification:
    try:
        structured_llm = _classifier_llm.with_structured_output(QueryClassification)
        messages = [
            SystemMessage(content=CLASSIFICATION_PROMPT),
            HumanMessage(content=query),
        ]
        result = structured_llm.invoke(messages)

        valid_categories = {"LOW_CHUNKS", "HIGH_CHUNKS", "GRAPH_RETRIEVAL"}
        if result.category not in valid_categories:
            result.category = "HIGH_CHUNKS"

        if result.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            result.category = "HIGH_CHUNKS"

        return result
    except Exception as e:
        print(f"Classification failed: {e}. Defaulting to HIGH_CHUNKS.")
        return QueryClassification(
            category="HIGH_CHUNKS",
            reason="Classification failed, using safe default",
            confidence=0.0,
        )
