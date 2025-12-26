"""Unit tests for BFL Flux MCP server."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bfl_flux_mcp.server import (
    BFLClient,
    _check_credits,
    _extract_image_url,
    _generate_image,
    get_client,
    list_tools,
)


class TestBFLClient:
    """Tests for BFLClient."""

    def test_init_requires_api_key(self):
        """Client requires API key."""
        client = BFLClient("test-key")
        assert client.api_key == "test-key"

    def test_headers_include_api_key(self):
        """Headers include x-key."""
        client = BFLClient("test-key")
        headers = client.client.headers
        assert headers["x-key"] == "test-key"
        assert headers["Content-Type"] == "application/json"


class TestGetClient:
    """Tests for get_client factory."""

    def test_raises_without_env_var(self):
        """Raises ValueError if BFL_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("BFL_API_KEY", None)
            with pytest.raises(ValueError, match="BFL_API_KEY"):
                get_client()

    def test_returns_client_with_env_var(self):
        """Returns client when env var is set."""
        with patch.dict(os.environ, {"BFL_API_KEY": "test-key"}):
            client = get_client()
            assert isinstance(client, BFLClient)
            assert client.api_key == "test-key"


class TestExtractImageUrl:
    """Tests for _extract_image_url helper."""

    def test_extracts_from_dict_with_sample(self):
        """Extracts URL from dict result."""
        result = {"result": {"sample": "https://example.com/image.png"}}
        assert _extract_image_url(result) == "https://example.com/image.png"

    def test_extracts_from_list_of_strings(self):
        """Extracts URL from list result."""
        result = {"result": ["https://example.com/image.png"]}
        assert _extract_image_url(result) == "https://example.com/image.png"

    def test_extracts_from_list_of_dicts(self):
        """Extracts URL from list of dicts."""
        result = {"result": [{"sample": "https://example.com/image.png"}]}
        assert _extract_image_url(result) == "https://example.com/image.png"

    def test_raises_on_empty_result(self):
        """Raises on empty result."""
        with pytest.raises(ValueError, match="No image URL"):
            _extract_image_url({"result": {}})

    def test_raises_on_missing_result(self):
        """Raises on missing result key."""
        with pytest.raises(ValueError, match="No image URL"):
            _extract_image_url({})


class TestListTools:
    """Tests for list_tools."""

    @pytest.mark.asyncio
    async def test_returns_three_tools(self):
        """Returns all three tools."""
        tools = await list_tools()
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_tool_names(self):
        """Tools have correct names."""
        tools = await list_tools()
        names = [t.name for t in tools]
        assert "generate_image" in names
        assert "edit_image" in names
        assert "check_credits" in names

    @pytest.mark.asyncio
    async def test_generate_image_schema(self):
        """generate_image has correct schema."""
        tools = await list_tools()
        gen_tool = next(t for t in tools if t.name == "generate_image")

        schema = gen_tool.inputSchema
        assert schema["required"] == ["prompt"]

        props = schema["properties"]
        assert "prompt" in props
        assert "model" in props
        assert "aspect_ratio" in props
        assert "prompt_upsampling" in props
        assert "output_format" in props

    @pytest.mark.asyncio
    async def test_generate_image_models(self):
        """generate_image has correct models."""
        tools = await list_tools()
        gen_tool = next(t for t in tools if t.name == "generate_image")

        models = gen_tool.inputSchema["properties"]["model"]["enum"]
        assert "flux-pro-1.1" in models
        assert "flux-2-pro" in models
        assert "flux-2-flex" in models
        # Removed models:
        assert "flux-schnell" not in models  # Not available via API


class TestCheckCredits:
    """Tests for check_credits tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_credits(self):
        """Returns formatted credit balance."""
        mock_client = MagicMock()
        mock_client.get_credits = AsyncMock(return_value={"credits": 100.0})

        result = await _check_credits(mock_client)

        assert len(result) == 1
        assert "Credits: 100.00" in result[0].text
        assert "$1.00 USD" in result[0].text

    @pytest.mark.asyncio
    async def test_handles_zero_credits(self):
        """Handles zero credits correctly."""
        mock_client = MagicMock()
        mock_client.get_credits = AsyncMock(return_value={"credits": 0})

        result = await _check_credits(mock_client)

        assert "Credits: 0.00" in result[0].text
        assert "$0.00 USD" in result[0].text

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.get_credits = AsyncMock(side_effect=Exception("API Error"))

        result = await _check_credits(mock_client)

        assert "Error checking credits" in result[0].text


class TestGenerateImage:
    """Tests for generate_image tool."""

    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Basic image generation works."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        result = await _generate_image(mock_client, {"prompt": "A test image"})

        assert len(result) == 1
        assert "Image generated successfully" in result[0].text
        assert "https://example.com/img.png" in result[0].text

    @pytest.mark.asyncio
    async def test_uses_correct_endpoint_for_model(self):
        """Uses correct API endpoint for each model."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        # Test flux-2-flex uses correct endpoint
        await _generate_image(
            mock_client, {"prompt": "test", "model": "flux-2-flex", "guidance": 5.0, "steps": 25}
        )

        call_args = mock_client.submit.call_args
        assert call_args[0][0] == "v1/flux-2-flex"
        assert call_args[0][1]["guidance"] == 5.0
        assert call_args[0][1]["steps"] == 25

    @pytest.mark.asyncio
    async def test_prompt_upsampling_passed(self):
        """prompt_upsampling is passed to API."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(mock_client, {"prompt": "test", "prompt_upsampling": True})

        call_args = mock_client.submit.call_args
        assert call_args[0][1]["prompt_upsampling"] is True

    @pytest.mark.asyncio
    async def test_polling_url_passed_to_wait(self):
        """polling_url from submit is passed to wait_for_completion."""
        mock_client = MagicMock()
        polling_url = "https://api.eu2.bfl.ai/v1/get_result?id=task-123"
        mock_client.submit = AsyncMock(return_value={"id": "task-123", "polling_url": polling_url})
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(mock_client, {"prompt": "test"})

        # Verify polling_url was passed
        mock_client.wait_for_completion.assert_called_once_with("task-123", polling_url)

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(side_effect=Exception("API Error"))

        result = await _generate_image(mock_client, {"prompt": "test"})

        assert "Error" in result[0].text
