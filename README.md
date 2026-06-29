# Rursee-Schifffahrt Chatbot 🚢

RAG-based chatbot for Rursee-Schifffahrt — answers questions about prices, ships and routes on Rursee and Obersee using LangChain, ChromaDB and Claude API.

## Tech Stack

* **LangChain** — RAG pipeline
* **ChromaDB** — vector database for embeddings
* **HuggingFace** — local embeddings model
* **Anthropic Claude** — LLM for answer generation
* **FastAPI** — REST API

## How it works

User question
↓
HuggingFace embeddings → ChromaDB similarity search
↓
Relevant context chunks retrieved
↓
Claude generates answer based on context
↓
FastAPI returns JSON response

## Installation

pip install -r requirements.txt

## Usage

uvicorn 2:app --reload

Open http://127.0.0.1:8000/docs to test the API.

## Example

Request:
{
"question": "Was kostet die Rursee-Rundfahrt für eine Familie?"
}

Response:
{
"question": "Was kostet die Rursee-Rundfahrt für eine Familie?",
"answer": "Die Rursee-Rundfahrt kostet für eine Familie 41,00 €."
}

## API Endpoints

* GET / — health check
* POST /ask — ask a question

