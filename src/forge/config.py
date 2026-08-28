"""Local configuration. Construct Settings when needed; never print credentials."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# This repository uses an editable package installation for local development.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FORGE"
    llm_enabled: bool = False
    llm_model: str = ""
    api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    @property
    def model_configured(self) -> bool:
        return bool(
            self.llm_enabled
            and self.llm_model.strip()
            and self.api_key
            and self.api_key.get_secret_value().strip()
        )
