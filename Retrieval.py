import os
import re
import sys
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import ENABLE_GRAPH_RETRIEVAL, GRAPH_FALLBACK_K, check_neo4j_available
from query_classifier import classify_query

load_dotenv()


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(
    persist_directory="dbfinal/chroma_db",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)
collection_count = db._collection.count()

if collection_count == 0:
    print(
        "WARNING: No chunks are indexed in dbfinal/chroma_db yet. "
        "Run Ingestion_pipeline.py first and make sure PDF_PATH points to a valid PDF."
    )

graph_available = False
if ENABLE_GRAPH_RETRIEVAL:
    graph_available = check_neo4j_available()
    if graph_available:
        print("Neo4j connected. Graph retrieval enabled.")
    else:
        print("Neo4j unavailable. Graph queries will fall back to vector retrieval.")

chatHistory = []


def retrieve_chunks(query: str, k: int = 2) -> list:
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def extract_text_content(content):
    """Return plain text from a model response content payload."""

    if isinstance(content, str):
        return clean_model_output(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(item["text"])
                elif item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)

        return clean_model_output("".join(parts))

    return clean_model_output(str(content))


def clean_model_output(text):
    """Remove hidden reasoning blocks before model output is displayed or reused."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def format_chunk_provenance(chunk):
    """Format stored source metadata for display."""

    source_file = chunk.metadata.get("source_file", "Unknown source")
    source_pages = chunk.metadata.get("source_pages", [])
    source_elements_raw = chunk.metadata.get("source_elements", "[]")

    try:
        source_elements = json.loads(source_elements_raw)
    except Exception:
        source_elements = []

    page_text = (
        ", ".join(str(page) for page in source_pages) if source_pages else "Unknown"
    )

    provenance_lines = [f"    Source: {source_file}", f"    Pages: {page_text}"]

    if source_elements:
        first_element = source_elements[0]
        element_type = first_element.get("type", "Unknown")
        page_number = first_element.get("page_number", "Unknown")
        coordinates = first_element.get("coordinates")
        provenance_lines.append(
            f"    First element: {element_type} on page {page_number}"
        )
        if coordinates:
            provenance_lines.append(f"    Coordinates: {coordinates}")

    return "\n".join(provenance_lines)


def get_standalone_query(query, chat_history):
    """Rewrite a follow-up question into a standalone query using chat history."""

    if not chat_history:
        return query
    messages = [
        SystemMessage(
            content="Rewrite the user's latest question into a standalone search query using the conversation history. Keep it concise and only return the rewritten query."
        ),
    ]
    messages.extend(chat_history[-4:])
    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    standalone_query = extract_text_content(response.content)

    return standalone_query or query


def generate_final_answer(chunks, query, chat_history=None):
    """Generate final answer using retrieved document chunks."""

    try:
        prompt_text = f"""Based on the following documents, answer this question: {query}

IMPORTANT OUTPUT RULES:
- Return only the final answer for the user.
- Do not show or mention your reasoning, analysis, chain of thought, or thinking process.
- Never output <think> tags or any content inside them.

CONTENT TO ANALYZE:
"""

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text[:2000]}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html[:2]):
                        prompt_text += f"Table {j + 1}:\n{table[:500]}\n\n"
            else:
                prompt_text += f"TEXT:\n{chunk.page_content[:2000]}\n\n"

            prompt_text += "\n"

        prompt_text += """
Provide a clear, comprehensive answer. If the documents don't contain sufficient information, say so. Cite source documents when possible. Return only the final answer, never include reasoning or <think> tags.

ANSWER:"""

        if chat_history:
            history_text = "\n".join(
                f"{message.type.capitalize()}: {message.content[:200]}"
                for message in chat_history[-4:]
            )
            prompt_text = f"Conversation history:\n{history_text}\n\n{prompt_text}"

        print("Generating final answer...")
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])

        return extract_text_content(response.content)

    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."


def main():
    print("History-aware RAG ready. Type 'exit' or 'quit' to stop.\n")
    print(f"Loaded {collection_count} stored chunk(s) from Chroma.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not query:
            continue

        standalone_query = get_standalone_query(query, chatHistory)

        classification = classify_query(standalone_query)
        print(
            f"[Classification: {classification.category} "
            f"(confidence: {classification.confidence:.2f})]"
        )

        if classification.category == "GRAPH_RETRIEVAL" and graph_available:
            from graph_retrieval import perform_impact_analysis
            from impact_analysis import generate_impact_answer

            impact_result = perform_impact_analysis(standalone_query, llm)

            if impact_result.entities_found:
                supporting_chunks = retrieve_chunks(standalone_query, k=2)
                final_answer = generate_impact_answer(
                    impact_result, supporting_chunks, standalone_query, llm, chatHistory
                )
            else:
                print(
                    "  Entity not found in graph. Falling back to vector retrieval."
                )
                chunks = retrieve_chunks(standalone_query, k=GRAPH_FALLBACK_K)
                if not chunks:
                    print("Assistant: No relevant documents found.\n")
                    continue
                final_answer = generate_final_answer(
                    chunks, standalone_query, chatHistory
                )
        else:
            if classification.category == "GRAPH_RETRIEVAL" and not graph_available:
                print(
                    "  Neo4j unavailable. Falling back to vector retrieval (k=4)."
                )
                k = GRAPH_FALLBACK_K
            elif classification.category == "HIGH_CHUNKS":
                k = 4
            else:
                k = 2

            chunks = retrieve_chunks(standalone_query, k=k)

            if not chunks:
                print(
                    "Assistant: I couldn't retrieve any chunks because the vector store is empty. "
                    "Run the ingestion pipeline to populate dbfinal/chroma_db first.\n"
                )
                continue

            print(f"Retrieved {len(chunks)} chunk(s):")
            for index, chunk in enumerate(chunks, start=1):
                preview = chunk.page_content[:200].replace("\n", " ")
                print(f"  {index}. {preview}...")
                print(format_chunk_provenance(chunk))

            final_answer = generate_final_answer(
                chunks, standalone_query, chatHistory
            )

        chatHistory.append(HumanMessage(content=query))
        chatHistory.append(AIMessage(content=final_answer))
        print("=" * 150)
        print(f"Assistant: {final_answer}\n")


if __name__ == "__main__":
    main()
