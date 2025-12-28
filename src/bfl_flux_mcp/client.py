"""Async HTTP client for the Black Forest Labs API."""

import asyncio
import time
from typing import Any

import httpx

from .config import Settings, get_settings


class BFLClient:
    """Async client for the Black Forest Labs API."""

    def __init__(self, settings_or_api_key: Settings | str | None = None):
        """Initialize client with settings or API key.

        Args:
            settings_or_api_key: Settings instance, API key string, or None to load from env.
        """
        # Support backwards compatibility: accept string API key
        if isinstance(settings_or_api_key, str):
            # Create settings with just the API key
            self._api_key = settings_or_api_key
            self._api_base_url = "https://api.bfl.ai"
            self._credits_api_url = "https://api.bfl.ml/v1/credits"
            self._timeout = 30.0
            self._poll_interval = 2.0
            self._max_wait_time = 300
        else:
            settings = settings_or_api_key or get_settings()
            self._api_key = settings.api_key
            self._api_base_url = settings.api_base_url
            self._credits_api_url = settings.credits_api_url
            self._timeout = settings.timeout
            self._poll_interval = settings.poll_interval
            self._max_wait_time = settings.max_wait_time

        self.client = httpx.AsyncClient(
            base_url=self._api_base_url,
            headers={
                "x-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )

    @property
    def api_key(self) -> str:
        """Get the API key (for backwards compatibility)."""
        return self._api_key

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a generation request.

        Args:
            endpoint: API endpoint path (e.g., "v1/flux-pro-1.1")
            payload: Request payload

        Returns:
            Response containing task ID and optional polling URL
        """
        response = await self.client.post(f"/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_result(self, task_id: str, polling_url: str | None = None) -> dict[str, Any]:
        """Get the result of a task.

        Args:
            task_id: Task ID from submission
            polling_url: Optional polling URL (may be on different server)

        Returns:
            Task status and result
        """
        if polling_url:
            # Use the polling_url directly (may be on different server like api.eu2.bfl.ai)
            response = await self.client.get(
                polling_url,
                headers={"x-key": self._api_key},
            )
        else:
            response = await self.client.get(
                "/v1/get_result",
                params={"id": task_id},
                headers={"x-key": self._api_key},
            )
        response.raise_for_status()
        return response.json()

    async def wait_for_completion(
        self, task_id: str, polling_url: str | None = None
    ) -> dict[str, Any]:
        """Poll until task is complete.

        Args:
            task_id: Task ID from submission
            polling_url: Optional polling URL

        Returns:
            Completed task result

        Raises:
            Exception: If task fails or times out
        """
        start_time = time.time()

        while time.time() - start_time < self._max_wait_time:
            result = await self.get_result(task_id, polling_url)
            status = result.get("status")

            if status == "Ready":
                return result

            if status in ("Error", "Content Moderated", "Request Moderated"):
                raise Exception(f"Task failed: {status}")

            await asyncio.sleep(self._poll_interval)

        raise Exception("Task timed out")

    async def get_credits(self) -> dict[str, Any]:
        """Get account credit balance.

        Note: Credits endpoint is on api.bfl.ml, not api.bfl.ai

        Returns:
            Dict containing credits balance
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                self._credits_api_url,
                headers={"x-key": self._api_key},
            )
            response.raise_for_status()
            return response.json()

    async def list_finetunes(self) -> dict[str, Any]:
        """List all finetuned models for this account.

        Returns:
            Dict containing list of finetunes
        """
        response = await self.client.get(
            "/v1/my_finetunes",
            headers={"x-key": self._api_key},
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "BFLClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def get_client() -> BFLClient:
    """Get a new BFL client instance.

    Returns:
        Configured BFLClient

    Raises:
        ValueError: If BFL_API_KEY is not set
    """
    return BFLClient()
