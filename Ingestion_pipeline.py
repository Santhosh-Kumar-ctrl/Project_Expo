import os
from Chunking import partition_doc, chunking_by_title
from ingestion_with_images import summarise_chunks
from Vectorization import create_vector_store
from config import ENABLE_GRAPH_RETRIEVAL, check_neo4j_available
from dotenv import load_dotenv

load_dotenv()


def collect_pdf_paths_from_folder(folder_path):
    """Collect all PDF files from a folder recursively."""

    pdf_paths = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Docs folder not found: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file_name))

    return sorted(pdf_paths)


def normalize_pdf_paths(pdf_paths):
    if pdf_paths is None:
        return []

    if isinstance(pdf_paths, str):
        pdf_paths = pdf_paths.strip()

        if os.path.isdir(pdf_paths):
            return collect_pdf_paths_from_folder(pdf_paths)

        return [path.strip() for path in pdf_paths.split(",") if path.strip()]

    return [path for path in pdf_paths if path]


def advanced_ingestion_pipeline(pdf_paths):
    """Run the complete RAG ingestion pipeline"""
    normalized_paths = normalize_pdf_paths(pdf_paths)

    if not normalized_paths:
        raise ValueError(
            "No PDF paths were provided. Set PDF_PATH or PDF_PATHS to one or more valid PDF files."
        )

    print("Starting RAG Ingestion Pipeline")
    print("-+-" * 50)

    all_summarised_chunks = []

    for index, pdf_path in enumerate(normalized_paths, start=1):
        print(f"Processing document {index}/{len(normalized_paths)}: {pdf_path}")

        # Step 1: Partition
        elements = partition_doc(pdf_path)

        # Step 2: Chunk
        chunks = chunking_by_title(elements)

        # Step 3: AI Summarisation
        summarised_chunks = summarise_chunks(chunks)

        for doc in summarised_chunks:
            doc.metadata["source_file"] = pdf_path
            doc.metadata["source_index"] = index

        all_summarised_chunks.extend(summarised_chunks)

    if not all_summarised_chunks:
        raise ValueError("No chunks were produced from the provided PDF files.")

    # Step 4: Vector Store
    db = create_vector_store(
        all_summarised_chunks, persist_directory="dbfinal/chroma_db"
    )

    print(f"Indexed {db._collection.count()} chunk(s) in dbfinal/chroma_db.")

    if ENABLE_GRAPH_RETRIEVAL and check_neo4j_available():
        from graph_ingestion import run_graph_ingestion

        run_graph_ingestion(all_summarised_chunks)
    elif ENABLE_GRAPH_RETRIEVAL:
        print(
            "Neo4j unavailable. Skipping graph ingestion. "
            "Set up Neo4j and re-run to populate the knowledge graph."
        )

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    pdf_paths = os.getenv("PDF_PATHS") or os.getenv("PDF_PATH")

    if not pdf_paths:
        docs_folder = os.getenv("DOCS_FOLDER", "docs")
        pdf_paths = collect_pdf_paths_from_folder(docs_folder)

        if not pdf_paths:
            raise ValueError(
                f"No PDF files were found in '{docs_folder}'. Put your documents there and run ingestion again."
            )
    elif isinstance(pdf_paths, str) and os.path.isdir(pdf_paths):
        pdf_paths = collect_pdf_paths_from_folder(pdf_paths)

    advanced_ingestion_pipeline(pdf_paths=pdf_paths)
