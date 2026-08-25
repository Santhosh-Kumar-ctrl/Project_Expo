import json
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {"text": chunk.text, "tables": [], "images": [], "types": ["text"]}

    # Check for tables and images in original elements
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # Handle images
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def extract_chunk_provenance(chunk):
    """Collect source document, page, and location metadata for a chunk."""

    provenance = {
        "source_file": None,
        "pages": [],
        "elements": [],
    }

    if not hasattr(chunk, "metadata"):
        return provenance

    if hasattr(chunk.metadata, "filename"):
        provenance["source_file"] = chunk.metadata.filename

    if not hasattr(chunk.metadata, "orig_elements"):
        return provenance

    pages = set()

    for element_index, element in enumerate(chunk.metadata.orig_elements, start=1):
        element_metadata = getattr(element, "metadata", None)

        page_number = getattr(element_metadata, "page_number", None)

        coordinates = getattr(element_metadata, "coordinates", None)

        element_type = type(element).__name__

        if page_number is not None:
            pages.add(page_number)

        # Convert Unstructured CoordinatesMetadata
        # into JSON-serializable data.
        coordinates_data = None

        if coordinates is not None:
            try:
                coordinates_data = {
                    "points": getattr(coordinates, "points", None),
                    "system": str(getattr(coordinates, "system", None)),
                }
            except Exception:
                coordinates_data = str(coordinates)

        provenance["elements"].append(
            {
                "element_index": element_index,
                "type": element_type,
                "page_number": page_number,
                "coordinates": coordinates_data,
            }
        )

        if provenance["source_file"] is None:
            source_file = getattr(element_metadata, "filename", None)

            if source_file:
                provenance["source_file"] = source_file

    provenance["pages"] = sorted(pages)

    return provenance


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOllama(model="minicpm-v4.6", temperature=0)

        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n\n"

                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"     AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("Processing chunks with AI Summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")

        # Analyze chunk content
        content_data = separate_content_types(chunk)
        provenance = extract_chunk_provenance(chunk)

        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(
            f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}"
        )

        # Create AI-enhanced summary if chunk has tables/images
        if content_data["tables"] or content_data["images"]:
            print("     -> Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
                print("     -> AI summary created successfully")
                print(f"     -> Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     AI summary failed: {e}")
                enhanced_content = content_data["text"]
        else:
            print("     -> Using raw text (no tables/images)")
            enhanced_content = content_data["text"]

        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                ),
                "source_file": provenance["source_file"],
                "source_pages": provenance["pages"],
                "source_elements": json.dumps(provenance["elements"]),
                "chunk_index": current_chunk,
            },
        )

        langchain_documents.append(doc)

    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []

    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(
                    doc.metadata.get("original_content", "{}")
                ),
                "source_file": doc.metadata.get("source_file"),
                "source_pages": doc.metadata.get("source_pages", []),
                "source_elements": json.loads(
                    doc.metadata.get("source_elements", "[]")
                ),
                "chunk_index": doc.metadata.get("chunk_index"),
            },
        }
        export_data.append(chunk_data)

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(export_data)} chunks to {filename}")
    return export_data
