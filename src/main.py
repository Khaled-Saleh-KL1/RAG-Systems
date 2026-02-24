# Libraries Imports
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

# Files Imports
from helpers import get_settings
from routes import base_router, data_router, qa_router
from stores import ChromaDBStore

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Start ---
    settings = get_settings()

    # ChromaDB Database Connection
    chroma_store = ChromaDBStore()
    chroma_store.connect()
    app.state.chroma_store = chroma_store
    print("[Lifespan] ChromaDB connected.")

    yield

    # --- Close ---
    chroma_store.disconnect()
    print("[Lifespan] ChromaDB disconnected.")

app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(qa_router)

# Serve React GUI from views/
views_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")
app.mount("/app", StaticFiles(directory=views_dir, html=True), name="views")
