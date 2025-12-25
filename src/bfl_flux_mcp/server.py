#!/usr/bin/env python3
"""
BFL Flux MCP Server - Minimal Python implementation

Exposes two tools:
- generate_image: Text-to-image generation with model selection
- edit_image: Image editing using Kontext models

API Reference: https://docs.bfl.ml
"""

import asyncio
import os
import time
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# --- Configuration ---

API_BASE_URL = "https://api.bfl.ai"
POLL_INTERVAL = 2.0  # seconds
MAX_WAIT_TIME = 300  # 5 minutes


# --- API Client ---


class BFLClient:
    """Async client for the Black Forest Labs API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            headers={"x-key": api_key, "Content-Type": "application/json"},
            timeout=30.0,
        )

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a generation request."""
        response = await self.client.post(f"/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Get the result of a task."""
        response = await self.client.get(
            "/v1/get_result",
            params={"id": task_id},
            headers={"x-key": self.api_key},
        )
        response.raise_for_status()
        return response.json()

    async def wait_for_completion(self, task_id: str) -> dict[str, Any]:
        """Poll until task is complete."""
        start_time = time.time()

        while time.time() - start_time < MAX_WAIT_TIME:
            result = await self.get_result(task_id)
            status = result.get("status")

            if status == "Ready":
                return result

            if status in ("Error", "Content Moderated", "Request Moderated"):
                raise Exception(f"Task failed: {status}")

            await asyncio.sleep(POLL_INTERVAL)

        raise Exception("Task timed out")

    async def close(self):
        await self.client.aclose()


# --- MCP Server ---

server = Server("bfl-flux")


def get_client() -> BFLClient:
    """Get API client, raising helpful error if key is missing."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        raise ValueError(
            "BFL_API_KEY environment variable is required. " "Get your key at https://api.bfl.ml"
        )
    return BFLClient(api_key)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="generate_image",
            description=(
                "Generate an image from a text prompt using BFL Flux models. "
                "Returns the URL of the generated image (valid for 10 minutes). "
                "Models: flux-pro (best quality), flux-dev (fast), flux-pro-1.1 (balanced), "
                "flux-2-pro (latest), flux-2-dev (latest fast)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: flux-pro-1.1)",
                        "enum": [
                            "flux-pro",
                            "flux-dev",
                            "flux-pro-1.1",
                            "flux-pro-1.1-ultra",
                            "flux-2-pro",
                            "flux-2-dev",
                        ],
                        "default": "flux-pro-1.1",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio (default: 1:1)",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                        "default": "1:1",
                    },
                    "width": {
                        "type": "integer",
                        "description": (
                            "Width in pixels (multiple of 32, 256-1440). Overrides aspect_ratio."
                        ),
                    },
                    "height": {
                        "type": "integer",
                        "description": (
                            "Height in pixels (multiple of 32, 256-1440). Overrides aspect_ratio."
                        ),
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for reproducibility (optional)",
                    },
                    "safety_tolerance": {
                        "type": "integer",
                        "description": "Safety filter (0=strict, 6=permissive, default: 2)",
                        "minimum": 0,
                        "maximum": 6,
                        "default": 2,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="edit_image",
            description=(
                "Edit an existing image using natural language instructions. "
                "Provide either a base64-encoded image or a URL. "
                "Models: kontext-pro (balanced), kontext-max (highest quality), "
                "fill-pro (inpainting)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instructions for how to edit the image",
                    },
                    "image": {
                        "type": "string",
                        "description": "Base64-encoded image data or URL of the image to edit",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: kontext-pro)",
                        "enum": ["kontext-pro", "kontext-max", "fill-pro"],
                        "default": "kontext-pro",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Output aspect ratio (default: preserve original)",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for reproducibility (optional)",
                    },
                    "safety_tolerance": {
                        "type": "integer",
                        "description": (
                            "Safety filter (0=strict, 2=permissive for Kontext, default: 2)"
                        ),
                        "minimum": 0,
                        "maximum": 2,
                        "default": 2,
                    },
                },
                "required": ["prompt", "image"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool."""
    client = get_client()

    try:
        if name == "generate_image":
            return await _generate_image(client, arguments)
        elif name == "edit_image":
            return await _edit_image(client, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        await client.close()


async def _generate_image(client: BFLClient, args: dict[str, Any]) -> list[TextContent]:
    """Handle generate_image tool."""
    # Map model name to endpoint
    model_map = {
        "flux-pro": "v1/flux-pro",
        "flux-dev": "v1/flux-dev",
        "flux-pro-1.1": "v1/flux-pro-1.1",
        "flux-pro-1.1-ultra": "v1/flux-pro-1.1-ultra",
        "flux-2-pro": "v1/flux-2-pro",
        "flux-2-dev": "v1/flux-2-dev",
    }

    model = args.get("model", "flux-pro-1.1")
    endpoint = model_map.get(model, "v1/flux-pro-1.1")

    # Build payload
    payload: dict[str, Any] = {"prompt": args["prompt"]}

    # Handle dimensions
    if "width" in args and "height" in args:
        payload["width"] = args["width"]
        payload["height"] = args["height"]
    elif "aspect_ratio" in args:
        payload["aspect_ratio"] = args["aspect_ratio"]

    # Optional params
    if "seed" in args:
        payload["seed"] = args["seed"]
    if "safety_tolerance" in args:
        payload["safety_tolerance"] = args["safety_tolerance"]

    try:
        # Submit and wait
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]

        result = await client.wait_for_completion(task_id)

        # Extract image URL
        image_url = _extract_image_url(result)

        return [
            TextContent(
                type="text",
                text=(
                    f"Image generated successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Prompt:** {args['prompt'][:100]}{'...' if len(args['prompt']) > 100 else ''}\n"
                    f"**Image URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def _edit_image(client: BFLClient, args: dict[str, Any]) -> list[TextContent]:
    """Handle edit_image tool."""
    # Map model name to endpoint
    model_map = {
        "kontext-pro": "v1/flux-kontext-pro",
        "kontext-max": "v1/flux-kontext-max",
        "fill-pro": "v1/flux-fill-pro",
    }

    model = args.get("model", "kontext-pro")
    endpoint = model_map.get(model, "v1/flux-kontext-pro")

    # Build payload
    payload: dict[str, Any] = {
        "prompt": args["prompt"],
        "input_image": args["image"],  # Can be base64 or URL
    }

    # Optional params
    if "aspect_ratio" in args:
        payload["aspect_ratio"] = args["aspect_ratio"]
    if "seed" in args:
        payload["seed"] = args["seed"]
    if "safety_tolerance" in args:
        payload["safety_tolerance"] = args["safety_tolerance"]

    try:
        # Submit and wait
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]

        result = await client.wait_for_completion(task_id)

        # Extract image URL
        image_url = _extract_image_url(result)

        return [
            TextContent(
                type="text",
                text=(
                    f"Image edited successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Edit instruction:** {args['prompt'][:100]}{'...' if len(args['prompt']) > 100 else ''}\n"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


def _extract_image_url(result: dict[str, Any]) -> str:
    """Extract image URL from API result."""
    result_data = result.get("result", {})

    if isinstance(result_data, dict) and "sample" in result_data:
        return result_data["sample"]
    elif isinstance(result_data, list) and len(result_data) > 0:
        item = result_data[0]
        return item if isinstance(item, str) else item.get("sample", "")

    raise ValueError("No image URL found in result")


def main():
    """Run the MCP server."""
    asyncio.run(_run_server())


async def _run_server():
    """Async server runner."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
