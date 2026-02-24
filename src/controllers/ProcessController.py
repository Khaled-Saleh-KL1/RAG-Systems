# Libraries Imports
import os
import time
import pymupdf
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

# Files Imports
from .BaseController import BaseController
from models import ProcessingEnums


class ProcessController(BaseController):
    def __init__(self, project_path: str):
        super().__init__()
        self.project_path = project_path

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        file_path = os.path.join(self.project_path, file_id)
        file_extension = self.get_file_extension(file_id=file_id)

        if file_extension == ProcessingEnums.PDF.value:
            return ArabicTextPDFLoader(file_path=file_path)

        return None

    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)
        if loader is None:
            return []
        return loader.load()

    def process_file_content(self, file_content: list,
                             chunk_size: int = 100, overlap_size: int = 20):
        pass


class ArabicTextPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Extract Arabic text from text-based PDF"""
        print(f"[PDF] Opening: {self.file_path}")
        t0 = time.time()

        text = self._extract_with_pymupdf()

        if not text.strip():
            print("[PDF] PyMuPDF returned empty, falling back to LangChain...")
            text = self._extract_with_langchain()

        elapsed = time.time() - t0
        preview = text[:100].replace('\n', ' ') if text else '(empty)'
        print(f"[PDF] Done in {elapsed:.1f}s — {len(text)} chars — preview: {preview}")

        return [Document(
            page_content=text,
            metadata={
                "source": self.file_path,
                "type": "text-based_pdf"
            }
        )]

    def _extract_with_pymupdf(self):
        """Extract text using PyMuPDF directly (better Arabic support)"""
        try:
            doc = pymupdf.open(self.file_path)
            total_pages = len(doc)
            print(f"[PDF] Document has {total_pages} page(s)")
            text = ""
            for i, page in enumerate(doc):
                page_text = page.get_text("text", sort=True)
                text += page_text + "\n"
                print(f"[PDF]   Page {i+1}/{total_pages} — {len(page_text)} chars")
            doc.close()
            return text
        except Exception as e:
            print(f"[PDF] PyMuPDF extraction failed: {e}")
            return ""

    def _extract_with_langchain(self):
        """Fallback to LangChain's PyMuPDFLoader"""
        try:
            loader = PyMuPDFLoader(self.file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[PDF] LangChain extraction failed: {e}")
            return ""
