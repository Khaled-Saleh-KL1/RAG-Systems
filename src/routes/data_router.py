# Libraries Imports
from fastapi import APIRouter, Depends, UploadFile, File,\
BackgroundTasks, Form, Request
from typing import List

# Files Imports
from helpers import Settings, get_settings
from controllers import ProjectController, ProcessController, DataController
from models import ResponseSignal
from stores import ChromaDBStore

data_router = APIRouter(
    prefix="/data",
    include_in_schema=True
)

@data_router.post("/upload/")
async def upload_files(
        request: Request,
        background_tasks: BackgroundTasks,
        project_id: str = Form(...),
        files: List[UploadFile] = File(...)
):
    project_ctrl = ProjectController()
    project_path = project_ctrl.get_project_path(project_id=project_id)

    file_ids = []
    for file in files:
        file_location = f"{project_path}/{file.filename}"
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        file_ids.append(file.filename)

    chroma_store = request.app.state.chroma_store
    background_tasks.add_task(run_ocr_pipeline, project_path, file_ids, project_id, chroma_store)

    return {
        "status": ResponseSignal.FILE_UPLOAD_SUCCESSFULLY,
        "message": f"Uploaded {len(files)} files. Processing started in background.",
        "project_id": project_id
    }

def run_ocr_pipeline(project_path: str, file_ids: List[str], project_id: str, chroma_store: ChromaDBStore):
    process_ctrl = ProcessController(project_path=project_path)
    data_ctrl = DataController(chroma_store=chroma_store)

    for file_id in file_ids:
        print(f"\n[Pipeline] Processing: {file_id}")
        content = process_ctrl.get_file_content(file_id=file_id)

        if content:
            data_ctrl.process_and_store(documents=content, project_id=project_id)
        else:
            print(f"[Pipeline] No content extracted from: {file_id}")

    print(f"[Pipeline] All files processed for project: {project_id}")
