# Libraries Imports
from fastapi import APIRouter, Request
from pydantic import BaseModel
import uuid

# Files Imports
from helpers import get_settings
from stores import GeminiLLM

qa_router = APIRouter(
    prefix="/qa",
    include_in_schema=True
)

NO_ANSWER_PHRASE = "لا أستطيع العثور على إجابة"


class QuestionRequest(BaseModel):
    question: str


class TeachRequest(BaseModel):
    question: str
    answer: str


@qa_router.post("/ask")
async def ask_question(request: Request, body: QuestionRequest):
    settings = get_settings()
    chroma_store = request.app.state.chroma_store

    # Search ChromaDB for relevant context
    documents = chroma_store.search(query=body.question, n_results=5)
    context = "\n\n".join(documents) if documents else ""

    if not context:
        return {
            "answer": "لا توجد بيانات في قاعدة البيانات بعد. يرجى رفع ملفات PDF أولاً.",
            "context_found": False,
            "knows_answer": False
        }

    # Ask Gemini with context
    llm = GeminiLLM(
        api_key=settings.GEMINI_API,
        model_name=settings.GEMINI_MODEL
    )
    answer = llm.answer_question(question=body.question, context=context)

    knows = NO_ANSWER_PHRASE not in answer

    return {
        "answer": answer,
        "context_found": True,
        "knows_answer": knows
    }


@qa_router.post("/teach")
async def teach_answer(request: Request, body: TeachRequest):
    chroma_store = request.app.state.chroma_store

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

    print(f"[QA] Human taught: {body.question[:60]} → {body.answer[:60]}")

    return {
        "status": "stored",
        "message": "تم حفظ الإجابة بنجاح. سأتذكرها في المرة القادمة."
    }
