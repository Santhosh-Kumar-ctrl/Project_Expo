import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


import json

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
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

retriever = db.as_retriever(search_kwargs={"k": 2})
chatHistory = []


def extract_text_content(content):
    """Return plain text from a model response content payload."""

    if isinstance(content, str):
        return content.strip()

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

        return "".join(parts).strip()

    return str(content).strip()


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
    """Generate final answer using multimodal content"""

    try:
        # Initialize LLM (needs vision model for images)

        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j + 1}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        if chat_history:
            prompt_text = (
                "Conversation history:\n"
                + "\n".join(
                    f"{message.type.capitalize()}: {message.content}"
                    for message in chat_history[-4:]
                )
                + "\n\n"
                + prompt_text
            )

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                # print(f"Original data: {original_data}")  # Debugging line
                images_base64 = original_data.get("images_base64", [])

                for image_base64 in images_base64:
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )

        # Send to AI and get response
        print("Generating final answer...")
        print(prompt_text)  # Debugging line
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return extract_text_content(response.content)

    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
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
        chunks = retriever.invoke(standalone_query)

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

        final_answer = generate_final_answer(chunks, standalone_query, chatHistory)

        chatHistory.append(HumanMessage(content=query))
        chatHistory.append(AIMessage(content=final_answer))
        print("=" * 150)
        print(f"Assistant: {final_answer}\n")


if __name__ == "__main__":
    main()
