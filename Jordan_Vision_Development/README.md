# 🇯🇴 Jordan Vision — مساعد البيانات

An AI-powered **Retrieval-Augmented Generation (RAG)** system built to answer questions about **Jordan's national vision documents**. Upload Arabic PDF files, and the system will extract, clean, store, and let you query them in natural language through a ChatGPT-like web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Data Flow](#data-flow)
- [Sample Data](#sample-data)

---

## Overview

**Jordan Vision** is a full-stack RAG application designed for Arabic document analysis. It allows users to:

1. **Upload PDF documents** (e.g., government strategy papers, policy documents)
2. **Automatically extract and clean Arabic text** from PDFs
3. **Store document embeddings** in a vector database (ChromaDB)
4. **Ask questions in natural language** and receive answers sourced from the uploaded documents
5. **Maintain personalized, persistent chat history** with a Claude-style conversation sidebar

The system uses **Google Gemini** as the LLM for both text cleaning and question answering, and **ChromaDB** for semantic search through document embeddings.

---

## Features

### 🔍 RAG Pipeline
- **PDF text extraction** using PyMuPDF with LangChain fallback
- **Arabic text cleaning** via Gemini (fixes RTL issues, disjointed letters, reconstructs tables)
- **Vector storage** with ChromaDB for semantic search
- **Context-aware Q&A** — answers are strictly grounded in document content

### 💬 Chat Interface & GuardRails
- Modern, dark-themed **React-based** web UI (RTL/Arabic-native layout)
- **Claude-style slide-out sidebar** with conversation history
- **Personalized sessions** — each browser gets its own user identity via `localStorage`
- **Multiple conversations** — create, switch between, and delete conversations
- **Auto-titled conversations** — named after the first question asked
- **Conversation memory** — the LLM receives the last 20 messages as context so it remembers what was discussed
- **AI GuardRails** — the system classifies questions to ensure responses are relevant:
  - If a question is **unrelated** to the documents, it informs the user without taking further action.
  - If a question is **related but missing an exact answer**, it prompts the user to teach it.
  - **Multilingual Support**: The GuardRails are designed to process the internal logic efficiently but will always reply and interact with the user in the **same language** they asked the question in (e.g., asking in English yields an English response).

### 📝 Teaching System
- When the LLM can't find an answer to a related question, a **teach box** appears
- Users can provide the correct answer, which gets stored in ChromaDB for future retrieval

### 📎 File Upload
- Drag and upload PDF files through the web interface
- Background processing — the UI remains responsive while files are processed

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                          │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │ Sidebar  │  │  Chat Container  │  │    Input Area         │ │
│  │ (History)│  │  (Messages)      │  │    (Fixed Bottom)     │ │
│  └──────────┘  └──────────────────┘  └───────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (REST API)
┌──────────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ base_router │  │  qa_router   │  │     data_router        │ │
│  │  GET /      │  │  POST /ask   │  │     POST /upload       │ │
│  │             │  │  POST /teach │  │                        │ │
│  │             │  │  CRUD /conv  │  │                        │ │
│  │             │  │  GET /hist   │  │                        │ │
│  └─────────────┘  └──────┬───────┘  └────────┬───────────────┘ │
│                          │                    │                 │
│               ┌──────────▼────────┐  ┌───────▼───────────────┐ │
│               │    Controllers    │  │    Controllers         │ │
│               │  (Business Logic) │  │  DataController        │ │
│               │                   │  │  ProcessController     │ │
│               │                   │  │  ProjectController     │ │
│               └───────┬───────────┘  └───────┬───────────────┘ │
│                       │                      │                 │
│         ┌─────────────▼──────────────────────▼───────────┐     │
│         │                  Stores                        │     │
│         │  ┌──────────┐  ┌────────────┐  ┌────────────┐  │     │
│         │  │ GeminiLLM│  │ ChromaDB   │  │ ChatHistory│  │     │
│         │  │ (Gemini  │  │ (Vector DB)│  │ (SQLite)   │  │     │
│         │  │  API)    │  │            │  │            │  │     │
│         │  └──────────┘  └────────────┘  └────────────┘  │     │
│         └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Jordan_Vision_Development/
├── Data/                          # Sample PDF documents
│   ├── Citizens_version.pdf
│   ├── Data_emerging_tech.pdf
│   ├── Jordan_on_the_Energy_map.pdf
│   ├── Ministry_of_Energy_Strategies.pdf
│   ├── jordan_vision_first_phase.pdf
│   └── jordan_vision_second_phase.pdf
├── docker/                        # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/                           # Application source code
│   ├── main.py                    # FastAPI entry point & lifespan
│   ├── .env                       # Environment variables (secrets)
│   ├── .example.env               # Example env template
│   ├── requirements.txt           # Python dependencies
│   │
│   ├── routes/                    # API route definitions
│   │   ├── base_router.py         # GET / — health check
│   │   ├── data_router.py         # POST /data/upload — file upload
│   │   └── qa_router.py           # Q&A, teach, conversations, history
│   │
│   ├── controllers/               # Business logic
│   │   ├── BaseController.py      # Base class with settings & paths
│   │   ├── ProjectController.py   # Project directory management
│   │   ├── ProcessController.py   # PDF text extraction (PyMuPDF)
│   │   └── DataController.py      # Text cleaning & ChromaDB storage
│   │
│   ├── stores/                    # Data storage layer
│   │   ├── llm/                   # LLM abstraction
│   │   │   ├── LLMInterface.py    # Abstract interface
│   │   │   └── GeminiLLM.py       # Google Gemini implementation
│   │   ├── ChromaDB/              # Vector database
│   │   │   ├── ChromaDBStore.py   # ChromaDB operations
│   │   │   └── data/              # Persisted ChromaDB data
│   │   └── ChatHistory/           # Chat history database
│   │       ├── ChatHistoryStore.py# SQLite-backed history & conversations
│   │       └── data/              # SQLite database file
│   │
│   ├── models/                    # Data models & enums
│   │   └── enums/
│   │       ├── ResponseEnum.py    # API response signals
│   │       ├── AssetTypeEnum.py   # Asset type definitions
│   │       └── ProcessingEnums.py # File processing enums
│   │
│   ├── helpers/                   # Utility modules
│   │   └── config.py             # Pydantic settings (reads .env)
│   │
│   ├── views/                     # Frontend
│   │   └── index.html             # React SPA (inline JSX + CSS)
│   │
│   ├── assets/files/              # Uploaded files storage
│   └── output/                    # Cleaned text output for verification
│
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 (via CDN + Babel) | Single-page chat interface |
| **Backend** | FastAPI (Python) | REST API server |
| **LLM** | Google Gemini (`gemini-2.5-flash`) | Text cleaning & Q&A |
| **Vector DB** | ChromaDB | Document embeddings & semantic search |
| **Chat DB** | SQLite | Conversation history & session management |
| **PDF Extraction** | PyMuPDF + LangChain | Arabic PDF text extraction |
| **Config** | Pydantic Settings | Environment variable management |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Google Gemini API key** ([Get one here](https://ai.google.dev/))
- **pip** (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/Khaled-Saleh-KL1/RAG-Systems.git
cd RAG-Systems/Jordan_Vision_Development
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

**Additionally install** (if not already covered by requirements):
```bash
pip install pydantic-settings
```

### 4. Configure Environment Variables

```bash
cp .example.env .env
```

Edit `.env` and fill in your API keys:

```env
APP_NAME="Jordan Vision In The Future"
APP_VERSION="0.01"

GEMINI_MODEL="gemini-2.5-flash"
GEMINI_API="your-gemini-api-key-here"

HUGGINGFACE_TOKEN="your-hf-token-here"

FILE_ALLOWED_TYPE=".pdf"
```

> ⚠️ **Important:** Never commit your `.env` file with real API keys. The `.gitignore` is already configured to exclude it.

---

## Configuration

All configuration is managed through environment variables loaded via **Pydantic Settings** from the `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `APP_NAME` | Application name | `"Jordan Vision In The Future"` |
| `APP_VERSION` | App version | `"0.01"` |
| `GEMINI_MODEL` | Gemini model to use | `"gemini-2.5-flash"` |
| `GEMINI_API` | Google Gemini API key | `"AIza..."` |
| `HUGGINGFACE_TOKEN` | HuggingFace API token | `"hf_..."` |
| `FILE_ALLOWED_TYPE` | Allowed upload file types | `".pdf"` |

---

## Running the Application

### Option A: Using Docker (Recommended)

You can easily run the application using Docker Compose. Make sure your `.env` contains the `GEMINI_API` and `GEMINI_MODEL`.

```bash
docker-compose --project-directory docker up --build
```

The app will be available at `http://localhost:8000/app`.

### Option B: Local Setup

Start the Server manually:

```bash
cd src
python -m uvicorn main:app --reload --port 8000
```

### Access the Application

| URL | Description |
|-----|-------------|
| `http://localhost:8000/app` | 🖥️ **Web Chat Interface** |
| `http://localhost:8000/` | 📡 Health check / welcome endpoint |
| `http://localhost:8000/docs` | 📖 Swagger API documentation |

---

## Usage Guide

### 1. Upload PDF Documents

1. Open the web interface at `http://localhost:8000/app`
2. Click **"📄 رفع ملف PDF"** (Upload PDF) in the top header
3. Select one or more PDF files
4. A toast notification confirms upload — processing runs in the background

**What happens behind the scenes:**
- PDF text is extracted using PyMuPDF
- Arabic text is cleaned by Gemini (fixes RTL, spacing, and encoding issues)
- Cleaned text is saved to `output/` for verification
- Document is stored as embeddings in ChromaDB

### 2. Ask Questions

1. Type your question in the input box at the bottom
2. Press **Enter** or click **"إرسال"** (Send)
3. The system searches ChromaDB for relevant context, then asks Gemini to answer

### 3. Teach the System

When the LLM responds with "I can't find an answer":
1. A **teach box** appears below the response
2. Type the correct answer
3. Click **"💾 حفظ الإجابة"** (Save Answer) — the Q&A pair is stored in ChromaDB for future queries

### 4. Manage Conversations

- Click the **☰ hamburger icon** to open the conversation sidebar
- Click **"✨ محادثة جديدة"** to start a new conversation
- Click any past conversation to switch to it and load its messages
- Hover over a conversation and click **🗑** to delete it
- Conversations are auto-titled from the first question you ask

### 5. Personalization

- Each browser automatically gets a unique **user ID** (stored in `localStorage`)
- All conversations are tied to your user ID
- Different browsers/devices will have different chat histories
- Refreshing the page restores your last active conversation

---

## API Reference

### Base

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns app name and version |

### Q&A

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/qa/ask` | Ask a question (requires `question`, `conversation_id`, `user_id`) |
| `POST` | `/qa/teach` | Teach the system a Q&A pair (requires `question`, `answer`, `conversation_id`) |

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/qa/conversations` | Create a new conversation (requires `user_id`) |
| `GET` | `/qa/conversations/{user_id}` | List all conversations for a user |
| `DELETE` | `/qa/conversations/{conversation_id}` | Delete a conversation and its messages |

### Chat History

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/qa/history/{conversation_id}` | Get the last 100 messages of a conversation |

### File Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/data/upload/` | Upload PDF files (multipart form: `project_id` + `files`) |

---

## Data Flow

### Upload Pipeline

```
User uploads PDF
      │
      ▼
 data_router.py ──► saves file to assets/files/{project_id}/
      │
      ▼  (background task)
 ProcessController ──► PyMuPDF extracts Arabic text
      │
      ▼
 DataController ──► GeminiLLM.clean_text() fixes Arabic issues
      │
      ├──► saves cleaned text to output/ (for verification)
      │
      ▼
 ChromaDBStore.store_document() ──► embeddings stored in ChromaDB
```

### Q&A Pipeline

```
User sends question
      │
      ▼
 qa_router.py ──► ChromaDBStore.search() finds top 5 relevant docs
      │
      ▼
 ChatHistoryStore.get_history() ──► fetches last 20 messages
      │
      ▼
 GeminiLLM.answer_question()
      │ (receives: context + conversation history + question)
      │
      ▼
 Gemini generates answer grounded in the context
      │
      ├──► Response sent to user
      │
      ▼
 ChatHistoryStore.add_message() ──► saves Q&A to SQLite
```

---

## Sample Data

The `Data/` directory contains Jordan Vision national planning documents:

| Document | Description |
|----------|-------------|
| `jordan_vision_first_phase.pdf` | Jordan Vision 2025 — First Phase |
| `jordan_vision_second_phase.pdf` | Jordan Vision 2025 — Second Phase |
| `Citizens_version.pdf` | Citizens version of the national plan |
| `Ministry_of_Energy_Strategies.pdf` | Energy sector strategies |
| `Jordan_on_the_Energy_map.pdf` | Jordan's position on the global energy map |
| `Data_emerging_tech.pdf` | Emerging technology and data strategies |

Upload these through the web interface to start asking questions about Jordan's national development plans.

---

## Database Details

### ChromaDB (Vector Database)
- **Location:** `src/stores/ChromaDB/data/`
- **Collection name:** `documents`
- **Purpose:** Stores document embeddings for semantic search
- **Populated by:** Upload pipeline + teach feature

### SQLite (Chat History)
- **Location:** `src/stores/ChatHistory/data/chat_history.db`
- **Tables:**
  - `conversations` — conversation metadata (id, user_id, title, timestamps)
  - `chat_messages` — individual messages (conversation_id, role, content, timestamp)
- **Auto-migration:** On startup, if the old schema is detected it is automatically upgraded

---

## Repository
Find the source code and development details here: [Jordan Vision Development Repo](https://github.com/Khaled-Saleh-KL1/RAG-Systems/tree/Jordan/Jordan_Vision_Development).

## License

This project is developed for **HTU (Al-Hussein Technical University)** educational purposes.
