# Hybrid RAG + Knowledge Graph System

A policy document question-answering system that combines vector-based retrieval (RAG) with a Neo4j knowledge graph for impact analysis. The system intelligently classifies queries and routes them to the appropriate retrieval strategy.

## Features

- **Query Classification**: Automatically classifies queries into three categories using an LLM
  - `LOW_CHUNKS` - Simple factual queries (retrieves 2 chunks)
  - `HIGH_CHUNKS` - Complex queries requiring broader context (retrieves 4 chunks)
  - `GRAPH_RETRIEVAL` - What-if / impact analysis queries (uses Neo4j knowledge graph)
- **Knowledge Graph**: Neo4j-based graph representing policy entities, requirements, and their dependencies
- **Impact Analysis**: Identifies direct and indirect impacts of policy changes through graph traversal
- **Multimodal Ingestion**: Processes PDFs with OCR, table extraction, and image analysis
- **History-Aware**: Maintains conversation context for follow-up questions

## Architecture

```
User Query
    |
    v
Query Rewriting (history-aware)
    |
    v
Query Classification (Groq LLM)
    |
    +---> LOW_CHUNKS ---> ChromaDB k=2 ---> LLM Answer
    |
    +---> HIGH_CHUNKS ---> ChromaDB k=4 ---> LLM Answer
    |
    +---> GRAPH_RETRIEVAL ---> Neo4j Impact Analysis ---> LLM Impact Report
```

## Prerequisites

- Python 3.11+
- Docker (for Neo4j)
- Ollama (for local embeddings)
- Groq API key (free tier works)
- Tesseract OCR (for PDF processing)

## Installation

### 1. Clone and Set Up Virtual Environment

```bash
git clone <repository-url>
cd project_expo

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up the Unstructured Library

The `unstructured` library requires additional system dependencies for hi-res PDF processing.

#### Install Tesseract OCR

**Windows:**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add the install directory to your system PATH
4. Verify: `tesseract --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

**Mac:**
```bash
brew install tesseract
```

#### Install Additional Unstructured Dependencies

```bash
# For PDF processing with hi-res strategy
pip install "unstructured[pdf]"

# If you encounter issues with poppler (PDF rendering):
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
#          Extract and add the bin/ folder to PATH
# Linux: sudo apt-get install poppler-utils
# Mac: brew install poppler

# For table structure inference
pip install unstructured-inference
```

#### Verify Unstructured Installation

```bash
python -c "from unstructured.partition.pdf import partition_pdf; print('unstructured OK')"
```

### 4. Install and Configure Ollama

Ollama runs the local embedding model (`nomic-embed-text`).

1. Download from https://ollama.com/download
2. Install and start Ollama
3. Pull required models:

```bash
ollama pull nomic-embed-text
```

Verify Ollama is running:
```bash
ollama list
```

### 5. Set Up Neo4j (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Verify Neo4j is running:
- Open http://localhost:7474 in your browser
- Login with username `neo4j` and password `password`

#### Initialize the Graph Schema

Open the Neo4j Browser (http://localhost:7474) and run each statement from `neo4j_setup.cypher`:

```cypher
CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (n:Policy) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT requirement_name IF NOT EXISTS FOR (n:Requirement) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT role_name IF NOT EXISTS FOR (n:Role) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT process_name IF NOT EXISTS FOR (n:Process) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT consequence_name IF NOT EXISTS FOR (n:Consequence) REQUIRE n.name IS UNIQUE;

CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Policy|Requirement|Entity|Role|Process|Consequence)
ON EACH [n.name, n.description];
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```env
PDF_PATH = "C:\path\to\your\project\docs"
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-20b"

# Neo4j Connection
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
NEO4J_DATABASE="neo4j"

# Feature Flags
ENABLE_GRAPH_RETRIEVAL=true
GRAPH_FALLBACK_K=4
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.5
```

Get a free Groq API key at https://console.groq.com/keys

## Running the Application

### Step 1: Generate Sample Policy PDF (Optional)

If you don't have your own policy documents yet:

```bash
pip install fpdf2
python create_sample_pdf.py
```

This creates `docs/sample_policy.pdf` with sample academic policies.

### Step 2: Add Your Documents

Place your PDF documents in the `docs/` folder. The system processes all PDFs in this directory recursively.

### Step 3: Run the Ingestion Pipeline

This processes PDFs into vector embeddings (ChromaDB) and populates the knowledge graph (Neo4j):

```bash
python Ingestion_pipeline.py
```

The pipeline performs:
1. PDF partitioning (OCR, table/image extraction)
2. Title-based chunking
3. AI-enhanced summarization for multimodal content
4. Vector store creation (ChromaDB with nomic-embed-text)
5. Knowledge graph population (entity/relationship extraction via Groq)

### Step 4: Run the Retrieval System

```bash
python Retrieval.py
```

### Example Queries

**Simple factual query (LOW_CHUNKS):**
```
You: What is the minimum attendance requirement?
```

**Complex query (HIGH_CHUNKS):**
```
You: Explain all the eligibility requirements for appearing in the examination.
```

**Impact analysis query (GRAPH_RETRIEVAL):**
```
You: What if the attendance percentage requirement is changed from 70% to 75%?
```

Type `exit` or `quit` to stop.

## Project Structure

```
project_expo/
├── Retrieval.py              # Main application (query routing + answer generation)
├── Ingestion_pipeline.py     # Orchestrates document ingestion
├── Chunking.py               # PDF partitioning and chunking
├── ingestion_with_images.py  # AI-enhanced summarization
├── Vectorization.py          # ChromaDB vector store creation
├── config.py                 # Centralized configuration
├── query_classifier.py       # Query classification (LOW/HIGH/GRAPH)
├── graph_schema.py           # Pydantic models for graph entities
├── graph_ingestion.py        # Entity extraction and Neo4j population
├── graph_retrieval.py        # Neo4j queries and impact analysis
├── impact_analysis.py        # Impact answer generation
├── neo4j_setup.cypher        # Neo4j schema (constraints + indexes)
├── create_sample_pdf.py      # Generates sample policy PDF for testing
├── requirements.txt          # Python dependencies
├── .env                      # Configuration (not tracked by git)
└── docs/
    └── sample_policy.pdf     # Sample academic policy document
```

## Troubleshooting

### Ollama not running
```
Error: Connection refused on localhost:11434
```
Start Ollama: open the Ollama application or run `ollama serve`

### Neo4j not available
The system gracefully falls back to vector-only retrieval (k=4) when Neo4j is unavailable. You'll see:
```
Neo4j unavailable. Graph queries will fall back to vector retrieval.
```
Start Neo4j: `docker start neo4j`

### Groq rate limit
```
Error code: 413 - Request too large
```
The free tier has token limits. The system already trims context, but if you hit limits:
- Wait a minute and retry
- Upgrade to Groq Dev Tier for higher limits

### Tesseract not found
```
TesseractNotFoundError
```
Ensure Tesseract is installed and its directory is in your system PATH. Restart your terminal after adding to PATH.

### Unicode errors on Windows console
If you see encoding errors when printing answers, the system handles this automatically via UTF-8 stdout reconfiguration. If issues persist, run:
```bash
chcp 65001
python Retrieval.py
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM (Generation) | Groq Cloud API (openai/gpt-oss-20b) |
| LLM (Ingestion) | Local Ollama (minicpm-v) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB (cosine similarity, HNSW) |
| Knowledge Graph | Neo4j 5.x |
| PDF Processing | unstructured (hi-res strategy) |
| Framework | LangChain (core abstractions only) |
