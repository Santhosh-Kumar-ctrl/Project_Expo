import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from graph_schema import QueryClassification
from config import CLASSIFICATION_CONFIDENCE_THRESHOLD

load_dotenv()

_classifier_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_tokens=256,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

CLASSIFICATION_PROMPT = """Classify the query into one category:
- LOW_CHUNKS: Simple factual lookups needing 1-2 excerpts.
- HIGH_CHUNKS: Complex questions needing multiple sections.
- GRAPH_RETRIEVAL: What-if scenarios, impact analysis, or policy change questions.

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
