# Libraries Imports
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import uuid
import os

# Files Imports
from helpers import get_settings
from stores import GeminiLLM

qa_router = APIRouter(
    prefix="/qa",
    include_in_schema=True
)

# Static messages
STATIC_NO_DATA_MSG = "لا توجد بيانات في قاعدة البيانات بعد. يرجى رفع ملفات PDF أولاً."
STATIC_IRRELEVANT_MSG = "عذراً، هذا السؤال خارج نطاق الوثائق المتاحة في قاعدة البيانات."

# Distance threshold — ChromaDB returns L2 distances, lower = more relevant
# Adjusted to 1.2 to prevent completely irrelevant questions from triggering the LLM.
# 1.8 was too high, allowing off-topic questions to pass as "related".
RELATED_THRESHOLD = 0.7


class QuestionRequest(BaseModel):
    question: str
    conversation_id: str
    user_id: str


class TeachRequest(BaseModel):
    question: str
    answer: str
    conversation_id: str


class NewConversationRequest(BaseModel):
    user_id: str


# --- Conversation endpoints ---

@qa_router.post("/conversations")
async def create_conversation(request: Request, body: NewConversationRequest):
    chat_history_store = request.app.state.chat_history_store
    conv_id = uuid.uuid4().hex
    conv = chat_history_store.create_conversation(conv_id, body.user_id)
    return conv


@qa_router.get("/conversations/{user_id}")
async def list_conversations(request: Request, user_id: str):
    chat_history_store = request.app.state.chat_history_store
    convs = chat_history_store.list_conversations(user_id)
    return {"conversations": convs}


@qa_router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    chat_history_store = request.app.state.chat_history_store
    chat_history_store.delete_conversation(conversation_id)
    return {"status": "deleted"}


# --- Chat endpoints ---

@qa_router.post("/ask")
async def ask_question(request: Request, body: QuestionRequest):
    settings = get_settings()
    chroma_store = request.app.state.chroma_store
    chat_history_store = request.app.state.chat_history_store

    # Search ChromaDB for relevant context
    search_results = chroma_store.search(query=body.question, n_results=5)
    documents = search_results["documents"]
    metadatas = search_results["metadatas"]
    distances = search_results["distances"]

    # --- Level 1: Initial Check ---
    if not documents:
        chat_history_store.add_message(body.conversation_id, "user", body.question)
        chat_history_store.add_message(body.conversation_id, "bot", STATIC_NO_DATA_MSG)
        _auto_title(chat_history_store, body.conversation_id, body.question)
        return {
            "answer": STATIC_NO_DATA_MSG,
            "context_found": False,
            "knows_answer": True  # Don't show box
        }

    # --- Scenario 3: Irrelevant Question (Distance too high) ---
    # Check the best (lowest) distance to determine general relevance
    best_distance = min(distances) if distances else float('inf')
    
    if best_distance > RELATED_THRESHOLD:
        chat_history_store.add_message(body.conversation_id, "user", body.question)
        chat_history_store.add_message(body.conversation_id, "bot", STATIC_IRRELEVANT_MSG)
        _auto_title(chat_history_store, body.conversation_id, body.question)
        return {
            "answer": STATIC_IRRELEVANT_MSG,
            "context_found": False,
            "knows_answer": True  # Don't show box for irrelevant topics
        }

    # --- Scenarios 1 & 2: Question is Related ---
    relevant_docs = [doc for doc, dist in zip(documents, distances) if dist <= RELATED_THRESHOLD]
    relevant_metas = [meta for meta, dist in zip(metadatas, distances) if dist <= RELATED_THRESHOLD]
    
    context = "\n\n".join(relevant_docs)

    # Extract source names for references
    sources = []
    for meta in relevant_metas:
        source = meta.get("source", "")
        if source and source != "human_teaching":
            name = os.path.basename(source)
            sources.append(name)
        elif source == "human_teaching":
            sources.append("تعليم المستخدم")

    # Fetch conversation history
    history = chat_history_store.get_history(body.conversation_id, limit=20)

    # Enforce a strict rule for the LLM to use a specific keyword if it doesn't know the answer
    strict_prompt = (
        f"{body.question}\n\n"
        "ملاحظة للنظام: إذا لم تكن الإجابة المحددة على هذا السؤال موجودة في السياق المرفق، "
        "يجب عليك أن تبدأ إجابتك بكلمة '[غير_متوفر]' ثم تعتذر وتوضح أن المعلومات غير موجودة."
    )

    # Ask Gemini
    llm = GeminiLLM(
        api_key=settings.GEMINI_API,
        model_name=settings.GEMINI_MODEL
    )
    answer = llm.answer_question(
        question=strict_prompt,
        context=context,
        history=history,
        sources=sources
    )

    # Detect if LLM failed to find a specific answer based on the enforced tag
    if "[غير_متوفر]" in answer:
        knows = False # Trigger the injection box (Scenario 2)
        clean_answer = answer.replace("[غير_متوفر]", "").strip()
    else:
        knows = True # Answer found (Scenario 1)
        clean_answer = answer

    # Save to chat history
    chat_history_store.add_message(body.conversation_id, "user", body.question)
    chat_history_store.add_message(body.conversation_id, "bot", clean_answer)
    _auto_title(chat_history_store, body.conversation_id, body.question)

    return {
        "answer": clean_answer,
        "context_found": True,
        "knows_answer": knows  # If False, shows teach box
    }


@qa_router.post("/teach")
async def teach_answer(request: Request, body: TeachRequest):
    chroma_store = request.app.state.chroma_store
    chat_history_store = request.app.state.chat_history_store

    doc_id = f"teach_{uuid.uuid4().hex[:8]}"
    text = f"سؤال: {body.question}\nجواب: {body.answer}"

    chroma_store.store_document(
        doc_id=doc_id,
        text=text,
        metadata={
            "source": "human_teaching",
            "type": "qa_pair"
        }
    )

    teach_msg = f"📝 تم تعليم الإجابة: {body.answer}"
    chat_history_store.add_message(body.conversation_id, "bot", teach_msg)

    print(f"[QA] Human taught: {body.question[:60]} → {body.answer[:60]}")

    return {
        "status": "stored",
        "message": "تم حفظ الإجابة بنجاح. سأتذكرها في المرة القادمة."
    }


@qa_router.get("/history/{conversation_id}")
async def get_history(request: Request, conversation_id: str):
    chat_history_store = request.app.state.chat_history_store
    messages = chat_history_store.get_history(conversation_id, limit=100)
    return {"messages": messages}


def _auto_title(store, conversation_id: str, question: str):
    """Set conversation title to first question (truncated) if still default."""
    convs = store.conn.cursor()
    convs.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
    row = convs.fetchone()
    if row and row["title"] == "محادثة جديدة":
        title = question[:50] + ("..." if len(question) > 50 else "")
        store.update_conversation_title(conversation_id, title)
