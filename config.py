import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

PDF_PATH = os.getenv("PDF_PATH") or os.getenv("PDF_PATHS")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

ENABLE_GRAPH_RETRIEVAL = os.getenv("ENABLE_GRAPH_RETRIEVAL", "true").lower() == "true"
GRAPH_FALLBACK_K = int(os.getenv("GRAPH_FALLBACK_K", "4"))
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.5")
)


def check_neo4j_available():
    if not ENABLE_GRAPH_RETRIEVAL or not NEO4J_PASSWORD:
        return False
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False
