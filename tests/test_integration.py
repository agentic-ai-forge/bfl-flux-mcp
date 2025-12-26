"""Integration tests for BFL Flux MCP server.

These tests make REAL API calls and consume credits.
Only run manually with a valid BFL_API_KEY.

Usage:
    # Run all integration tests
    BFL_API_KEY=your-key uv run pytest tests/test_integration.py -v

    # Run only specific tests
    BFL_API_KEY=your-key uv run pytest tests/test_integration.py -v -k "test_check_credits"

    # Skip credit-consuming tests (only run free/cheap tests)
    BFL_API_KEY=your-key uv run pytest tests/test_integration.py -v -m "not expensive"
"""

import os

import pytest

from bfl_flux_mcp.server import (
    _check_credits,
    _expand_image,
    _generate_image,
    _list_finetunes,
    get_client,
)

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("BFL_API_KEY"),
    reason="BFL_API_KEY not set - skipping integration tests",
)


class TestIntegrationCredits:
    """Integration tests for credits and account verification."""

    @pytest.mark.asyncio
    async def test_check_credits_real_api(self):
        """Verify API key works and credits endpoint responds."""
        client = get_client()
        try:
            result = await _check_credits(client)
            assert len(result) == 1
            assert "Credits:" in result[0].text
            assert "USD" in result[0].text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_list_finetunes_real_api(self):
        """Verify finetunes endpoint responds (may be empty list)."""
        client = get_client()
        try:
            result = await _list_finetunes(client)
            assert len(result) == 1
            # Either "No finetuned models" or "Your Finetuned Models"
            assert "finetune" in result[0].text.lower() or "model" in result[0].text.lower()
        finally:
            await client.close()


class TestIntegrationGenerate:
    """Integration tests for image generation.

    WARNING: These tests consume credits!
    """

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_generate_image_flux_dev(self):
        """Generate an image with flux-dev (cheapest model)."""
        client = get_client()
        try:
            result = await _generate_image(
                client,
                {
                    "prompt": "A simple red circle on white background",
                    "model": "flux-dev",
                    "aspect_ratio": "1:1",
                },
            )
            assert len(result) == 1
            assert "Image generated successfully" in result[0].text
            assert "https://" in result[0].text  # Contains URL
        finally:
            await client.close()

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_generate_image_with_seed(self):
        """Generate an image with seed for reproducibility."""
        client = get_client()
        try:
            result = await _generate_image(
                client,
                {
                    "prompt": "A blue square",
                    "model": "flux-dev",
                    "seed": 42,
                },
            )
            assert "Image generated successfully" in result[0].text
        finally:
            await client.close()


class TestIntegrationExpand:
    """Integration tests for image expansion."""

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_expand_image_basic(self):
        """Test image expansion with minimal input."""
        client = get_client()
        try:
            # Using a tiny valid PNG
            tiny_png_base64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIA"
                "X8jx0gAAAABJRU5ErkJggg=="
            )
            result = await _expand_image(
                client,
                {
                    "image": tiny_png_base64,
                    "right": 64,
                },
            )
            assert len(result) == 1
        finally:
            await client.close()


class TestIntegrationClient:
    """Integration tests for BFLClient directly."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Verify client can be created with API key."""
        client = get_client()
        assert client is not None
        assert client.api_key == os.environ["BFL_API_KEY"]
        await client.close()

    @pytest.mark.asyncio
    async def test_get_credits_raw(self):
        """Test raw credits API call."""
        client = get_client()
        try:
            credits_data = await client.get_credits()
            assert "credits" in credits_data
            assert isinstance(credits_data["credits"], (int, float))
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_list_finetunes_raw(self):
        """Test raw finetunes API call.

        Note: This endpoint may return 500 if finetunes feature isn't enabled
        for the account. We test that the call completes without exception
        from our code (API errors are acceptable).
        """
        client = get_client()
        try:
            try:
                finetunes_data = await client.list_finetunes()
                # Response should have a finetunes key (may be empty list)
                assert "finetunes" in finetunes_data or isinstance(finetunes_data, list)
            except Exception as e:
                # API may return 500 if finetunes not enabled - that's OK
                assert "500" in str(e) or "finetune" in str(e).lower()
        finally:
            await client.close()
