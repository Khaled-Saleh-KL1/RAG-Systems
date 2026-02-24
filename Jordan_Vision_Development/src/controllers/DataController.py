# Libraries Imports
import os

# Files Imports
from .BaseController import BaseController
from stores import GeminiLLM, ChromaDBStore


class DataController(BaseController):
    def __init__(self, chroma_store: ChromaDBStore):
        super().__init__()
        self.chroma_store = chroma_store
        self.llm = GeminiLLM(
            api_key=self.app_settings.GEMINI_API,
            model_name=self.app_settings.GEMINI_MODEL
        )

    def process_and_store(self, documents: list, project_id: str):
        # Create output folder for verification
        output_dir = os.path.join(self.base_dir, "..", "output")
        os.makedirs(output_dir, exist_ok=True)

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            file_name = os.path.basename(source)
            doc_id = f"{project_id}_{file_name}"

            print(f"[DataController] Cleaning text for: {file_name}")
            cleaned_text = self.llm.clean_text(doc.page_content)

            # Save cleaned text to file for verification
            output_path = os.path.join(output_dir, f"cleaned_{doc_id}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"[DataController] Saved cleaned text to: {output_path}")

            # Print preview
            preview = cleaned_text[:200].replace('\n', ' ')
            print(f"[DataController] Preview: {preview}...")

            print(f"[DataController] Storing in ChromaDB: {doc_id}")
            self.chroma_store.store_document(
                doc_id=doc_id,
                text=cleaned_text,
                metadata={
                    "source": source,
                    "project_id": project_id,
                    "type": doc.metadata.get("type", "")
                }
            )

            print(f"[DataController] Done: {doc_id}")

