from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    GEMINI_MODEL: str
    GEMINI_API: str

    HUGGINGFACE_TOKEN: str

    FILE_ALLOWED_TYPE: str

    class Config(SettingsConfigDict):
        env_file=".env"

def get_settings() -> Settings:
    return Settings()
