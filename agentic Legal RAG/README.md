# Agentic Legal RAG

A sophisticated proof-of-concept legal retrieval and synthesis workflow using ChromaDB for vector storage and a graph-based orchestration model powered by LangGraph. This system implements an "agentic" approach where the workflow can make decisions, such as routing to web search when local documents are insufficient.

## Overview

The Agentic Legal RAG system is designed to answer legal questions by retrieving relevant precedents from a vectorized database of legal documents, reranking them for relevance, and synthesizing comprehensive legal opinions with proper citations. Unlike traditional linear RAG systems, this implementation uses conditional edges in a graph workflow to handle cases where no relevant local documents are found, potentially routing to external web searches for up-to-date legal information.

### Key Features

- **Graph-Based Orchestration**: Uses LangGraph for workflow management with conditional routing
- **Multi-Model Strategy**: Leverages Groq (Llama-3) for fast query analysis and Gemini (1.5 Pro/Flash) for synthesis
- **Parent-Child Chunking**: Implements hierarchical document chunking for better retrieval
- **Metadata Filtering**: Supports topic and year-based filtering for precise legal searches
- **FastAPI Integration**: Provides REST API for easy integration
- **PDF Ingestion**: Automatically processes and chunks PDF legal documents

## Architecture

The system follows a graph-based workflow with the following nodes:

### Workflow Nodes

1. **QueryAnalysisNode**
   - **Purpose**: Analyzes the user's legal query to extract key information and create a search plan
   - **Model**: Groq Llama-3 (llama3-8b-8192)
   - **Functionality**:
     - Cleans and refines the query
     - Extracts metadata filters (topics like housing, contract, employment; years)
     - Determines if web search is needed based on query analysis
   - **Output**: Updates state with `search_plan` containing refined query, filters, and web search flag

2. **RetrievalNode**
   - **Purpose**: Retrieves relevant legal documents from the vector store
   - **Tool**: Custom VectorSearchTool using ChromaDB
   - **Functionality**:
     - Performs semantic search using the refined query
     - Applies metadata filters (topic, year)
     - Uses parent-child relationship for retrieving full article contexts
     - If no documents found, sets `needs_web_search = True`
   - **Output**: Updates state with `retrieved_docs` list

3. **RerankerNode**
   - **Purpose**: Filters and ranks retrieved documents by relevance
   - **Functionality**:
     - Sorts documents by similarity score (descending)
     - Filters documents with score >= 0.08
     - Falls back to top 3 if no documents meet threshold
   - **Output**: Updates state with `relevant_precedents` list

4. **SynthesisNode**
   - **Purpose**: Generates comprehensive legal opinions with citations
   - **Model**: Google Gemini 1.5 Flash
   - **Functionality**:
     - Combines relevant precedents into context
     - Generates professional legal opinion
     - Includes key principles, caveats, and citations
     - Handles fallback if LLM fails
   - **Output**: Updates state with `legal_opinion` and `citations`

5. **WebSearchNode**
   - **Purpose**: Placeholder for external web search functionality
   - **Functionality**: Currently returns a message indicating web search would be performed
   - **Note**: Web search implementation is planned but not yet integrated

### Data Flow

```
User Query
    ↓
QueryAnalysisNode → search_plan
    ↓
RetrievalNode → retrieved_docs
    ↓
[Conditional: if docs found]
    ↓
RerankerNode → relevant_precedents
    ↓
SynthesisNode → legal_opinion + citations
    ↓
[Else: no docs found]
    ↓
WebSearchNode → web search message
```

### State Management

The workflow uses a typed state dictionary (`LegalState`) with the following fields:

- `initial_query`: Original user question
- `search_plan`: Structured query plan with filters and web search flag
- `retrieved_docs`: List of retrieved documents (with reducer for accumulation)
- `relevant_precedents`: Filtered and ranked relevant documents
- `legal_opinion`: Final synthesized answer
- `citations`: List of document references used

### Vector Store Architecture

- **Database**: ChromaDB with persistent storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Chunking Strategy**: Parent-child hierarchical chunks
  - Child chunks: ~400 words with 80-word overlap
  - Parent chunks: Full 1000-word articles/sections
- **Metadata**: Source PDF, page number, article reference

## Setup

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
cd /home/sifen/Desktop/agentic/RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` in the project root:

```bash
cp .env.example .env
```

Then fill in your real values:

```
GROQ_API_KEY=your_groq_api_key_here
GMINI_API_KEY=your_gemini_api_key_here
```

Get API keys from:
- [Groq Console](https://console.groq.com/)
- [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Command Line Interface

#### Basic Query

```bash
python main.py --query "What are tenant rights for eviction in 2024?"
```

#### Query with PDF Ingestion

```bash
python main.py --ingest data/legal_document.pdf --query "What does this statute say about rent control?"
```

#### Custom Vector Store Directory

```bash
python main.py --persist-dir ./custom_chroma_store --query "housing law precedent"
```

### FastAPI HTTP API

#### Start the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to use the built-in frontend.

#### API Endpoints

- `GET /health`: Health check endpoint
- `POST /query`: Main query endpoint
- `GET /`: Simple web UI for sending legal queries

#### Query Request Format

```json
{
  "query": "What are the legal requirements for landlord-tenant disputes?",
  "persist_dir": "chroma_store",
  "ingest_path": "data/new_law.pdf"
}
```

#### Query Response Format

```json
{
  "legal_opinion": "Based on the relevant legal precedents...",
  "citations": ["Housing Act 2023-Article 1", "Tenant Rights Code-page-5"],
  "needs_web_search": false
}
```

### Document Ingestion

The system supports automatic ingestion of PDF documents:

- Place PDFs in the `data/` folder for automatic ingestion
- Use `--ingest` flag for one-time ingestion
- Documents are chunked and embedded into ChromaDB

## Dependencies

## Dependencies

Key dependencies include:

- **LangGraph**: Graph-based workflow orchestration
- **ChromaDB**: Vector database for document storage
- **Sentence Transformers**: Embedding generation
- **Groq**: Fast LLM for query analysis
- **Google GenAI**: Advanced LLM for synthesis
- **PyPDF**: PDF text extraction
- **FastAPI**: REST API framework
- **Pydantic**: Data validation

## Testing

Run the basic API/frontend route tests with:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Deploying on Render

### Option 1: Render Blueprint

This repo includes `render.yaml`, so you can create a new Render service directly from your GitHub repo and let Render detect the blueprint automatically.

### Option 2: Trigger Deploys from GitHub Actions

A workflow is included at `.github/workflows/render-deploy.yml` and calls `scripts/render_deploy.sh`.

Set one of these secret configurations in GitHub:

1. `RENDER_DEPLOY_HOOK_URL` (recommended), or
2. both `RENDER_API_KEY` and `RENDER_SERVICE_ID`.

When you push to `main` (or run workflow manually), GitHub triggers a Render deployment.

You can also trigger a deployment manually:

```bash
./scripts/render_deploy.sh
```

### Render First-Deploy Checklist

- Copy env file: `cp .env.example .env` for local development.
- Ensure your app keys are set in Render service environment:
  - `GROQ_API_KEY`
  - `GMINI_API_KEY`
- For GitHub-triggered deploys, set repository secrets:
  - `RENDER_DEPLOY_HOOK_URL` (recommended), or
  - `RENDER_API_KEY` and `RENDER_SERVICE_ID`
- Confirm Render start command is:
  - `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Push to `main` and confirm the GitHub workflow `Deploy to Render` succeeds.

## Configuration

### Vector Store Settings

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 400 words with 80-word overlap
- **Similarity Threshold**: 0.08 for reranking

### LLM Settings

- **Query Analysis**: Temperature 0.1, max tokens 200
- **Synthesis**: Temperature 0.1, max output tokens 1000

## Limitations and Future Work

### Current Limitations

- Web search functionality is placeholder only
- Limited to English legal documents
- No advanced legal reasoning beyond retrieval + synthesis
- Basic metadata filtering (topic, year only)

### Planned Enhancements

- Integration with real web search APIs (Google, Bing)
- Support for multiple languages
- Advanced legal reasoning capabilities
- Integration with legal databases (Westlaw, LexisNexis)
- Fine-tuned legal domain models
- Multi-document comparison features

## Contributing

This is a proof-of-concept system. Contributions are welcome for:

- Implementing web search functionality
- Adding support for more document formats
- Improving chunking strategies
- Enhancing metadata extraction
- Adding more sophisticated legal reasoning

## License

This project is for educational and research purposes. Please ensure compliance with API terms of service and legal document usage rights.
