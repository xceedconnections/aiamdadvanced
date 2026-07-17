from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "OpenAMD Advanced"
    APP_VERSION: str = "2.1.2"
    DEBUG: bool = False

    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5432
    DB_NAME: str = "openamd"
    DB_USER: str = "openamd"
    DB_PASS: str = "Openaccount@123"

    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    JWT_SECRET: str = "OpenAMD-JWT-Secret-Change-In-Production-2024"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 12

    ADMIN_USER: str = "admin"
    ADMIN_PASS: str = "Openaccount@123"
    ADMIN_EMAIL: str = "admin@openamd.local"

    RECORDINGS_DIR: str = "/opt/openamd/recordings"
    MODELS_DIR: str = "/opt/openamd/models"
    LOGS_DIR: str = "/opt/openamd/logs"

    ANALYSIS_SECONDS: float = 3.0
    MAX_AUDIO_MB: int = 5

    class Config:
        env_file = (".env", "/opt/openamd/backend/.env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def database_url(self) -> str:
        # URL-encode password so characters like @ do not break the DSN
        user = quote_plus(self.DB_USER)
        password = quote_plus(self.DB_PASS)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
