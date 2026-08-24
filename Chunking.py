import os
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

load_dotenv()


def partition_doc(file_path):
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    print(f"Partitioned {file_path} into {len(elements)} elements.")
    return elements


def chunking_by_title(elements):
    # Use the chunk_by_title function to chunk the elements by title
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2000,
        combine_text_under_n_chars=500,
    )
    print(f"Chunked into {len(chunks)} chunks.")
    return chunks
