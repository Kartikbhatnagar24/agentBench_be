# RAG System - Backend

This is the backend API for the Retrieval-Augmented Generation (RAG) application. It is powered by **FastAPI**, **LangChain**, **Qdrant** for vector search, and agentic workflows using **Google Gemini** and **Groq**. It also integrates with **Supabase** for user authentication and chat session storage, and **Ragas** for scoring/evaluation.

## Tech Stack
- **Framework**: FastAPI (Python 3.10+)
- **LLM/Embeddings**: Google Gemini, Groq, Hugging Face (Sentence Transformers)
- **Vector DB**: Qdrant
- **Database & Auth**: Supabase
- **Pipeline & Agents**: LangChain, PyMuPDF (fitz)
- **Evaluation**: Ragas, LangSmith

---

## Getting Started

### Prerequisites
- Python 3.10 or higher installed on your system.
- Access to Supabase, Qdrant, Google Gemini, and optionally Groq and LangSmith accounts.

### 1. Clone and Navigate
Clone the backend repository and navigate to the directory:
```bash
git clone <YOUR_BACKEND_GITHUB_REPO_URL> rag-backend
cd rag-backend
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment:
- **Windows (PowerShell/CMD):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Install all the required Python libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root of the `backend` directory:
```env
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_or_service_key

# Model & API Keys
GEMINI_API_KEY=your_google_gemini_api_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key

# Vector Database (Qdrant)
QDRANT_API=your_qdrant_api_key
QDRANT_ENDPOINT=your_qdrant_cluster_endpoint

# Evaluation and Tracing (Optional)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your_project_name

# Parsers
PDF_PARSER=fitz
```

### 5. Run the Server
Start the development server with **Uvicorn**:
```bash
uvicorn main:app --reload
```

Once running, the backend API will be available at `http://127.0.0.1:8000`. 
You can view the interactive Swagger API documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Directory Structure
- `agents/`: Contains LLM agent models, prompt templates, and agent pipeline logic (Fact Checking, Synthesis, Retrieval, and Scoring).
- `api/`: API router endpoints, schemas, and service logic for authentication, analysis, upload, chat sessions, and evaluations.
- `database/`: Database configuration and helpers.
- `eval/`: Evaluator module for running scoring metrics (Ragas).
- `rag_pipeline/`: Main RAG steps including text chunking, document embedding, and Qdrant ingestion.
- `utils/`: Miscellaneous utility functions and helper classes.
- `main.py`: Entrypoint for the FastAPI application.
