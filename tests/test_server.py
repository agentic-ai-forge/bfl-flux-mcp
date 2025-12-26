"""Unit tests for BFL Flux MCP server."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bfl_flux_mcp.server import (
    BFLClient,
    _check_credits,
    _edit_image,
    _extract_image_url,
    _generate_image,
    call_tool,
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

    @pytest.mark.asyncio
    async def test_submit_posts_to_endpoint(self):
        """Submit posts to the correct endpoint."""
        client = BFLClient("test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task-123"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.submit("v1/flux-pro-1.1", {"prompt": "test"})

        mock_post.assert_called_once_with("/v1/flux-pro-1.1", json={"prompt": "test"})
        assert result == {"id": "task-123"}

    @pytest.mark.asyncio
    async def test_get_result_without_polling_url(self):
        """get_result uses default endpoint when no polling_url."""
        client = BFLClient("test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "Ready"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.get_result("task-123")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["params"] == {"id": "task-123"}
        assert result == {"status": "Ready"}

    @pytest.mark.asyncio
    async def test_get_result_with_polling_url(self):
        """get_result uses polling_url when provided."""
        client = BFLClient("test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "Ready"}
        mock_response.raise_for_status = MagicMock()

        polling_url = "https://api.eu2.bfl.ai/v1/get_result?id=task-123"
        with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.get_result("task-123", polling_url)

        mock_get.assert_called_once_with(polling_url, headers={"x-key": "test-key"})
        assert result == {"status": "Ready"}

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_ready(self):
        """wait_for_completion returns when status is Ready."""
        client = BFLClient("test-key")

        with patch.object(client, "get_result", new_callable=AsyncMock) as mock_get_result:
            mock_get_result.return_value = {"status": "Ready", "result": {"sample": "url"}}
            result = await client.wait_for_completion("task-123")

        assert result["status"] == "Ready"

    @pytest.mark.asyncio
    async def test_wait_for_completion_raises_on_error(self):
        """wait_for_completion raises on error status."""
        client = BFLClient("test-key")

        with patch.object(client, "get_result", new_callable=AsyncMock) as mock_get_result:
            mock_get_result.return_value = {"status": "Error"}
            with pytest.raises(Exception, match="Task failed: Error"):
                await client.wait_for_completion("task-123")

    @pytest.mark.asyncio
    async def test_wait_for_completion_raises_on_content_moderated(self):
        """wait_for_completion raises on Content Moderated status."""
        client = BFLClient("test-key")

        with patch.object(client, "get_result", new_callable=AsyncMock) as mock_get_result:
            mock_get_result.return_value = {"status": "Content Moderated"}
            with pytest.raises(Exception, match="Task failed: Content Moderated"):
                await client.wait_for_completion("task-123")

    @pytest.mark.asyncio
    async def test_get_credits(self):
        """get_credits returns credit balance."""
        client = BFLClient("test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"credits": 100.0}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.get_credits()

        assert result == {"credits": 100.0}

    @pytest.mark.asyncio
    async def test_close(self):
        """close closes the httpx client."""
        client = BFLClient("test-key")

        with patch.object(client.client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()

        mock_close.assert_called_once()


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
            return_value={
                "id": "task-123",
                "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123",
            }
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
            return_value={
                "id": "task-123",
                "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123",
            }
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
            return_value={
                "id": "task-123",
                "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123",
            }
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

    @pytest.mark.asyncio
    async def test_width_height_override_aspect_ratio(self):
        """width/height override aspect_ratio."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(
            mock_client,
            {"prompt": "test", "width": 1024, "height": 768, "aspect_ratio": "1:1"},
        )

        call_args = mock_client.submit.call_args
        payload = call_args[0][1]
        assert payload["width"] == 1024
        assert payload["height"] == 768
        assert "aspect_ratio" not in payload

    @pytest.mark.asyncio
    async def test_aspect_ratio_passed(self):
        """aspect_ratio is passed when no width/height."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(mock_client, {"prompt": "test", "aspect_ratio": "16:9"})

        call_args = mock_client.submit.call_args
        assert call_args[0][1]["aspect_ratio"] == "16:9"

    @pytest.mark.asyncio
    async def test_optional_params_passed(self):
        """Optional params are passed to API."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(
            mock_client,
            {
                "prompt": "test",
                "seed": 42,
                "safety_tolerance": 4,
                "output_format": "jpeg",
            },
        )

        call_args = mock_client.submit.call_args
        payload = call_args[0][1]
        assert payload["seed"] == 42
        assert payload["safety_tolerance"] == 4
        assert payload["output_format"] == "jpeg"

    @pytest.mark.asyncio
    async def test_ultra_raw_mode(self):
        """raw mode for flux-pro-1.1-ultra."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _generate_image(
            mock_client, {"prompt": "test", "model": "flux-pro-1.1-ultra", "raw": True}
        )

        call_args = mock_client.submit.call_args
        assert call_args[0][0] == "v1/flux-pro-1.1-ultra"
        assert call_args[0][1]["raw"] is True

    @pytest.mark.asyncio
    async def test_long_prompt_truncated_in_output(self):
        """Long prompts are truncated in output message."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        long_prompt = "A" * 150
        result = await _generate_image(mock_client, {"prompt": long_prompt})

        assert "..." in result[0].text
        assert "A" * 100 in result[0].text

    @pytest.mark.asyncio
    async def test_credits_shown_in_output(self):
        """Credits used are shown in output when available."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={
                "status": "Ready",
                "result": {"sample": "https://example.com/img.png"},
                "cost": 4,
            }
        )

        result = await _generate_image(mock_client, {"prompt": "test"})

        assert "**Credits used:** 4" in result[0].text
        assert "$0.04" in result[0].text


class TestEditImage:
    """Tests for edit_image tool."""

    @pytest.mark.asyncio
    async def test_basic_edit(self):
        """Basic image editing works."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={
                "id": "task-123",
                "polling_url": "https://api.eu2.bfl.ai/v1/get_result?id=task-123",
            }
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/edited.png"}}
        )

        result = await _edit_image(mock_client, {"prompt": "Add a hat", "image": "base64data"})

        assert len(result) == 1
        assert "Image edited successfully" in result[0].text
        assert "https://example.com/edited.png" in result[0].text

    @pytest.mark.asyncio
    async def test_uses_correct_endpoint_for_model(self):
        """Uses correct API endpoint for each edit model."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _edit_image(mock_client, {"prompt": "test", "image": "data", "model": "kontext-max"})

        call_args = mock_client.submit.call_args
        assert call_args[0][0] == "v1/flux-kontext-max"

    @pytest.mark.asyncio
    async def test_fill_pro_endpoint(self):
        """fill-pro uses correct endpoint."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _edit_image(mock_client, {"prompt": "test", "image": "data", "model": "fill-pro"})

        call_args = mock_client.submit.call_args
        assert call_args[0][0] == "v1/flux-fill-pro"

    @pytest.mark.asyncio
    async def test_optional_params_passed(self):
        """Optional params are passed to edit API."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _edit_image(
            mock_client,
            {
                "prompt": "test",
                "image": "data",
                "aspect_ratio": "16:9",
                "seed": 42,
                "safety_tolerance": 1,
            },
        )

        call_args = mock_client.submit.call_args
        payload = call_args[0][1]
        assert payload["aspect_ratio"] == "16:9"
        assert payload["seed"] == 42
        assert payload["safety_tolerance"] == 1

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(side_effect=Exception("API Error"))

        result = await _edit_image(mock_client, {"prompt": "test", "image": "data"})

        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_polling_url_passed(self):
        """polling_url is passed to wait_for_completion."""
        mock_client = MagicMock()
        polling_url = "https://api.eu2.bfl.ai/v1/get_result?id=task-123"
        mock_client.submit = AsyncMock(return_value={"id": "task-123", "polling_url": polling_url})
        mock_client.wait_for_completion = AsyncMock(
            return_value={"status": "Ready", "result": {"sample": "https://example.com/img.png"}}
        )

        await _edit_image(mock_client, {"prompt": "test", "image": "data"})

        mock_client.wait_for_completion.assert_called_once_with("task-123", polling_url)

    @pytest.mark.asyncio
    async def test_credits_shown_in_output(self):
        """Credits used are shown in edit output when available."""
        mock_client = MagicMock()
        mock_client.submit = AsyncMock(
            return_value={"id": "task-123", "polling_url": "https://api.bfl.ai/v1/get_result"}
        )
        mock_client.wait_for_completion = AsyncMock(
            return_value={
                "status": "Ready",
                "result": {"sample": "https://example.com/img.png"},
                "cost": 8,
            }
        )

        result = await _edit_image(mock_client, {"prompt": "test", "image": "data"})

        assert "**Credits used:** 8" in result[0].text
        assert "$0.08" in result[0].text


class TestCallTool:
    """Tests for call_tool dispatcher."""

    @pytest.mark.asyncio
    async def test_dispatches_generate_image(self):
        """Dispatches to generate_image tool."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.submit = AsyncMock(
                return_value={"id": "task-123", "polling_url": "https://api.bfl.ai"}
            )
            mock_client.wait_for_completion = AsyncMock(
                return_value={
                    "status": "Ready",
                    "result": {"sample": "https://example.com/img.png"},
                }
            )
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await call_tool("generate_image", {"prompt": "test"})

            assert "Image generated successfully" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatches_edit_image(self):
        """Dispatches to edit_image tool."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.submit = AsyncMock(
                return_value={"id": "task-123", "polling_url": "https://api.bfl.ai"}
            )
            mock_client.wait_for_completion = AsyncMock(
                return_value={
                    "status": "Ready",
                    "result": {"sample": "https://example.com/img.png"},
                }
            )
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await call_tool("edit_image", {"prompt": "test", "image": "data"})

            assert "Image edited successfully" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatches_check_credits(self):
        """Dispatches to check_credits tool."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_credits = AsyncMock(return_value={"credits": 50.0})
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await call_tool("check_credits", {})

            assert "Credits: 50.00" in result[0].text

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Returns error for unknown tool."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await call_tool("unknown_tool", {})

            assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_closes_client_on_success(self):
        """Client is closed after successful call."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_credits = AsyncMock(return_value={"credits": 50.0})
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            await call_tool("check_credits", {})

            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_client_on_error(self):
        """Client is closed even when tool raises error."""
        with patch("bfl_flux_mcp.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.submit = AsyncMock(side_effect=Exception("Error"))
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            await call_tool("generate_image", {"prompt": "test"})

            mock_client.close.assert_called_once()
