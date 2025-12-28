"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """BFL Flux MCP Server settings.

    All settings can be configured via environment variables with BFL_ prefix.
    Example: BFL_API_KEY, BFL_POLL_INTERVAL, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="BFL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required
    api_key: str

    # API Configuration
    api_base_url: str = "https://api.bfl.ai"
    credits_api_url: str = "https://api.bfl.ml/v1/credits"

    # Polling Configuration
    poll_interval: float = 2.0  # seconds
    max_wait_time: int = 300  # 5 minutes

    # HTTP Client Configuration
    timeout: float = 30.0


def get_settings() -> Settings:
    """Get settings instance, raising helpful error if API key is missing."""
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as e:
        if "api_key" in str(e).lower():
            raise ValueError(
                "BFL_API_KEY environment variable is required. "
                "Get your key at https://api.bfl.ml"
            ) from e
        raise
