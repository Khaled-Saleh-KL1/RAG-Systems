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

    def answer_question(self, question: str, context: str, history: list[dict] = None, sources: list[str] = None) -> str:
        # Build conversation history section
        history_section = ""
        if history:
            history_lines = []
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role_label}: {msg['content']}")
            history_section = (
                "--- CONVERSATION HISTORY ---\n"
                + "\n".join(history_lines)
                + "\n--- END CONVERSATION HISTORY ---\n\n"
            )

        # Build sources section
        sources_section = ""
        if sources:
            unique_sources = list(dict.fromkeys(sources))  # deduplicate, keep order
            sources_section = "Available sources: " + ", ".join(unique_sources) + "\n"

        prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Answer in the same language as the question. "
            "You may also reference the conversation history if the user asks about previous questions or answers.\n"
            "At the end of your answer, list the source documents you used under a '📚 المراجع:' heading.\n\n"
            "--- CONTEXT START ---\n"
            f"{context}\n"
            "--- CONTEXT END ---\n\n"
            f"{sources_section}"
            f"{history_section}"
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
