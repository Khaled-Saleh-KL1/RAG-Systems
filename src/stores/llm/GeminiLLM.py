from google import genai
from .LLMInterface import LLMInterface


class GeminiLLM(LLMInterface):
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def clean_text(self, text: str) -> str:
        prompt = (
            "You are an Arabic text post-processor. "
            "The following text was extracted from a PDF and may contain errors. "
            "Your task:\n"
            "1. Fix disjointed letters, incorrect spacing, and overlapping words.\n"
            "2. Correct Right-to-Left (RTL) reading order issues (e.g., reversed sentences).\n"
            "3. Reconstruct tables and lists if the data implies a tabular format or bullet points.\n"
            "4. Do NOT hallucinate or add external information. Rely strictly on the provided text.\n"
            "5. Return ONLY the cleaned text, no explanations or commentary.\n\n"
            "--- RAW TEXT START ---\n"
            f"{text}\n"
            "--- RAW TEXT END ---"
        )

        print(f"[Gemini] Sending {len(text)} chars for cleaning...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        cleaned = response.text.strip()
        print(f"[Gemini] Received {len(cleaned)} cleaned chars.")
        return cleaned

    def answer_question(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant that answers questions ONLY based on the provided context. "
            "If the answer is not found in the context, say 'لا أستطيع العثور على إجابة في البيانات المتاحة' "
            "(I cannot find an answer in the available data). "
            "NEVER use external knowledge. Answer in the same language as the question.\n\n"
            "--- CONTEXT START ---\n"
            f"{context}\n"
            "--- CONTEXT END ---\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        print(f"[Gemini] Answering question: {question[:80]}...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        answer = response.text.strip()
        print(f"[Gemini] Answer: {answer[:100]}...")
        return answer
