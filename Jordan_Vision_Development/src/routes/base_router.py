# Libraries Imports
from fastapi import APIRouter, Depends

# Files Imports
from helpers import Settings, get_settings

base_router = APIRouter(include_in_schema=True)

@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "message": f"Welcome to {app_name} version {app_version}"
    }
