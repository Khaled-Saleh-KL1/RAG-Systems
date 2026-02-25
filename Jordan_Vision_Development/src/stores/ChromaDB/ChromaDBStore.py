import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from helpers import get_settings


class GeminiGenAIEmbeddingFunction(EmbeddingFunction):
    """
    Custom Embedding Function using the modern google.genai package 
    instead of the deprecated google.generativeai.
    """
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=input,
        )
        return [e.values for e in response.embeddings]


class ChromaDBStore:
    def __init__(self):
        self.client = None
        self.collection = None

    def connect(self, persist_dir: str = None):
        if persist_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            persist_dir = os.path.join(base, "data")

        settings = get_settings()
        google_ef = GeminiGenAIEmbeddingFunction(api_key=settings.GEMINI_API)

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=google_ef
        )
        print(f"[ChromaDB] Connected — persist dir: {persist_dir} (Using Gemini Embeddings)")

    def store_document(self, doc_id: str, text: str, metadata: dict = None):
        if self.collection is None:
            raise RuntimeError("ChromaDB not connected. Call connect() first.")

        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}]
        )
        print(f"[ChromaDB] Stored document: {doc_id} ({len(text)} chars)")

    def search(self, query: str, n_results: int = 5) -> dict:
        if self.collection is None:
            raise RuntimeError("ChromaDB not connected. Call connect() first.")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        print(f"[ChromaDB] Search returned {len(documents)} result(s), distances: {distances}")
        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }

    def disconnect(self):
        self.client = None
        self.collection = None
        print("[ChromaDB] Disconnected.")
