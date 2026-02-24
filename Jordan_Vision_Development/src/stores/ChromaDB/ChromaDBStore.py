import os
import chromadb


class ChromaDBStore:
    def __init__(self):
        self.client = None
        self.collection = None

    def connect(self, persist_dir: str = None):
        if persist_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            persist_dir = os.path.join(base, "data")

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="documents")
        print(f"[ChromaDB] Connected — persist dir: {persist_dir}")

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
