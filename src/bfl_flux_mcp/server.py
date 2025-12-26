#!/usr/bin/env python3
"""
BFL Flux MCP Server - Minimal Python implementation

Exposes seven tools:
- generate_image: Text-to-image generation with model selection
- edit_image: Image editing using Kontext models
- expand_image: Directional outpainting
- create_variation: Create variations using Redux (image_prompt)
- save_image: Download and save images before URLs expire
- check_credits: Verify API key and credit balance
- list_finetunes: View custom finetuned models

API Reference: https://docs.bfl.ml
"""

import asyncio
import base64
import os
import time
from pathlib import Path
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

    async def get_result(self, task_id: str, polling_url: str | None = None) -> dict[str, Any]:
        """Get the result of a task."""
        if polling_url:
            # Use the polling_url directly (may be on different server like api.eu2.bfl.ai)
            response = await self.client.get(
                polling_url,
                headers={"x-key": self.api_key},
            )
        else:
            response = await self.client.get(
                "/v1/get_result",
                params={"id": task_id},
                headers={"x-key": self.api_key},
            )
        response.raise_for_status()
        return response.json()

    async def wait_for_completion(
        self, task_id: str, polling_url: str | None = None
    ) -> dict[str, Any]:
        """Poll until task is complete."""
        start_time = time.time()

        while time.time() - start_time < MAX_WAIT_TIME:
            result = await self.get_result(task_id, polling_url)
            status = result.get("status")

            if status == "Ready":
                return result

            if status in ("Error", "Content Moderated", "Request Moderated"):
                raise Exception(f"Task failed: {status}")

            await asyncio.sleep(POLL_INTERVAL)

        raise Exception("Task timed out")

    async def get_credits(self) -> dict[str, Any]:
        """Get account credit balance.

        Note: Credits endpoint is on api.bfl.ml, not api.bfl.ai
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.bfl.ml/v1/credits",
                headers={"x-key": self.api_key},
            )
            response.raise_for_status()
            return response.json()

    async def list_finetunes(self) -> dict[str, Any]:
        """List all finetuned models for this account."""
        response = await self.client.get(
            "/v1/my_finetunes",
            headers={"x-key": self.api_key},
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


# --- MCP Server ---

server = Server("bfl-flux")


def get_client() -> BFLClient:
    """Get API client, raising helpful error if key is missing."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        raise ValueError(
            "BFL_API_KEY environment variable is required. Get your key at https://api.bfl.ml"
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
                "Models: flux-schnell (free/fast), flux-dev (free/good), flux-pro-1.1 (best), "
                "flux-2-pro (latest), flux-2-flex (adjustable steps/guidance)."
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
                        "description": (
                            "Model to use (default: flux-pro-1.1). "
                            "Costs: pro=$0.04, pro-1.1=$0.04, ultra=$0.06"
                        ),
                        "enum": [
                            "flux-dev",
                            "flux-pro",
                            "flux-pro-1.1",
                            "flux-pro-1.1-ultra",
                            "flux-2-pro",
                            "flux-2-flex",
                            "flux-2-max",
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
                            "Width in pixels (multiple of 16, max 2048). Overrides aspect_ratio."
                        ),
                    },
                    "height": {
                        "type": "integer",
                        "description": (
                            "Height in pixels (multiple of 16, max 2048). Overrides aspect_ratio."
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
                    "prompt_upsampling": {
                        "type": "boolean",
                        "description": "Enhance prompt for richer output (recommended for logos)",
                        "default": False,
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format (default: png)",
                        "enum": ["png", "jpeg"],
                        "default": "png",
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Prompt adherence 1.5-10 (flux-2-flex only)",
                        "minimum": 1.5,
                        "maximum": 10,
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Generation steps 1-50 (flux-2-flex only)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Raw mode for flux-pro-1.1-ultra",
                        "default": False,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="check_credits",
            description=(
                "Check your BFL API credit balance. Returns current credits available. "
                "1 credit = $0.01 USD. Use this to verify your account is configured correctly."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="edit_image",
            description=(
                "Edit an existing image using natural language instructions. "
                "Provide a local file path, URL, or base64-encoded image. "
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
                        "description": (
                            "Image to edit: local file path, URL, or base64-encoded data"
                        ),
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
        Tool(
            name="save_image",
            description=(
                "Download and save a generated image to a local file. "
                "Use this to save images before URLs expire (10 min). "
                "Supports PNG and JPEG formats."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the image to download (from generate/edit result)",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Destination file path (e.g., /path/to/image.png). "
                            "Parent directory must exist."
                        ),
                    },
                },
                "required": ["url", "path"],
            },
        ),
        Tool(
            name="expand_image",
            description=(
                "Expand an image by adding pixels to any side (outpainting). "
                "Specify pixels to add to top, bottom, left, and/or right. "
                "The AI will generate content that seamlessly extends the image."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": (
                            "Image to expand: local file path, URL, or base64-encoded data"
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional text description to guide the expansion",
                    },
                    "top": {
                        "type": "integer",
                        "description": "Pixels to add to the top (0-2048)",
                        "minimum": 0,
                        "maximum": 2048,
                        "default": 0,
                    },
                    "bottom": {
                        "type": "integer",
                        "description": "Pixels to add to the bottom (0-2048)",
                        "minimum": 0,
                        "maximum": 2048,
                        "default": 0,
                    },
                    "left": {
                        "type": "integer",
                        "description": "Pixels to add to the left (0-2048)",
                        "minimum": 0,
                        "maximum": 2048,
                        "default": 0,
                    },
                    "right": {
                        "type": "integer",
                        "description": "Pixels to add to the right (0-2048)",
                        "minimum": 0,
                        "maximum": 2048,
                        "default": 0,
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
                    "output_format": {
                        "type": "string",
                        "description": "Output format (default: png)",
                        "enum": ["png", "jpeg"],
                        "default": "png",
                    },
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="list_finetunes",
            description=(
                "List all your finetuned models. Shows model names, status, "
                "and identifiers that can be used with finetuned endpoints."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="create_variation",
            description=(
                "Create variations of an existing image using Redux. "
                "Generates images that maintain the essence of the original "
                "while applying optional text-guided modifications."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": (
                            "Source image: local file path, URL, or base64-encoded data"
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional text guidance for the variation",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: flux-pro-1.1)",
                        "enum": ["flux-pro-1.1", "flux-pro-1.1-ultra", "flux-dev"],
                        "default": "flux-pro-1.1",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio (default: 1:1)",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                        "default": "1:1",
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
                    "output_format": {
                        "type": "string",
                        "description": "Output format (default: png)",
                        "enum": ["png", "jpeg"],
                        "default": "png",
                    },
                },
                "required": ["image"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool."""
    # save_image doesn't need BFL client
    if name == "save_image":
        return await _save_image(arguments)

    client = get_client()

    try:
        if name == "generate_image":
            return await _generate_image(client, arguments)
        elif name == "edit_image":
            return await _edit_image(client, arguments)
        elif name == "check_credits":
            return await _check_credits(client)
        elif name == "expand_image":
            return await _expand_image(client, arguments)
        elif name == "list_finetunes":
            return await _list_finetunes(client)
        elif name == "create_variation":
            return await _create_variation(client, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        await client.close()


async def _check_credits(client: BFLClient) -> list[TextContent]:
    """Check BFL API credit balance."""
    try:
        result = await client.get_credits()
        credits = result.get("credits", 0)
        usd_value = credits * 0.01

        return [
            TextContent(
                type="text",
                text=(
                    f"**BFL API Credit Balance**\n\n"
                    f"Credits: {credits:.2f}\n"
                    f"Value: ${usd_value:.2f} USD\n\n"
                    f"Pricing reference:\n"
                    f"- flux-pro-1.1: 4 credits ($0.04)\n"
                    f"- flux-pro-1.1-ultra: 6 credits ($0.06)\n"
                    f"- kontext-pro: 4 credits ($0.04)\n"
                    f"- kontext-max: 8 credits ($0.08)"
                ),
            )
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error checking credits: {e!s}")]


async def _save_image(args: dict[str, Any]) -> list[TextContent]:
    """Download and save an image to a local file."""
    url = args["url"]
    path = Path(args["path"])

    try:
        # Validate path
        if not path.parent.exists():
            return [
                TextContent(
                    type="text",
                    text=f"Error: Parent directory does not exist: {path.parent}",
                )
            ]

        # Download image
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Get content type for validation
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Error: URL does not point to an image (content-type: {content_type})"
                        ),
                    )
                ]

            # Write to file
            path.write_bytes(response.content)

        file_size = path.stat().st_size
        size_kb = file_size / 1024

        return [
            TextContent(
                type="text",
                text=(f"Image saved successfully!\n\n**Path:** {path}\n**Size:** {size_kb:.1f} KB"),
            )
        ]

    except httpx.HTTPStatusError as e:
        return [TextContent(type="text", text=f"Error downloading image: {e!s}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error saving image: {e!s}")]


async def _generate_image(client: BFLClient, args: dict[str, Any]) -> list[TextContent]:
    """Handle generate_image tool."""
    # Map model name to endpoint
    model_map = {
        "flux-dev": "v1/flux-dev",
        "flux-pro": "v1/flux-pro",
        "flux-pro-1.1": "v1/flux-pro-1.1",
        "flux-pro-1.1-ultra": "v1/flux-pro-1.1-ultra",
        "flux-2-pro": "v1/flux-2-pro",
        "flux-2-flex": "v1/flux-2-flex",
        "flux-2-max": "v1/flux-2-max",
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
    if "prompt_upsampling" in args:
        payload["prompt_upsampling"] = args["prompt_upsampling"]
    if "output_format" in args:
        payload["output_format"] = args["output_format"]

    # Model-specific params
    if model == "flux-2-flex":
        if "guidance" in args:
            payload["guidance"] = args["guidance"]
        if "steps" in args:
            payload["steps"] = args["steps"]
    if model == "flux-pro-1.1-ultra" and args.get("raw"):
        payload["raw"] = True

    try:
        # Submit and wait
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        # Extract image URL and cost
        image_url = _extract_image_url(result)
        cost = result.get("cost")

        prompt_display = args["prompt"][:100]
        prompt_suffix = "..." if len(args["prompt"]) > 100 else ""

        cost_info = ""
        if cost is not None:
            usd = cost * 0.01
            cost_info = f"**Credits used:** {cost} (${usd:.2f})\n"

        return [
            TextContent(
                type="text",
                text=(
                    f"Image generated successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Prompt:** {prompt_display}{prompt_suffix}\n"
                    f"{cost_info}"
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

    # Handle image input: local file path, URL, or base64
    image_input = args["image"]
    if _is_local_file(image_input):
        try:
            image_input = _encode_image_to_base64(image_input)
        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"Error: {e!s}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading image file: {e!s}")]

    # Build payload
    payload: dict[str, Any] = {
        "prompt": args["prompt"],
        "input_image": image_input,
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
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        # Extract image URL and cost
        image_url = _extract_image_url(result)
        cost = result.get("cost")

        prompt_display = args["prompt"][:100]
        prompt_suffix = "..." if len(args["prompt"]) > 100 else ""

        cost_info = ""
        if cost is not None:
            usd = cost * 0.01
            cost_info = f"**Credits used:** {cost} (${usd:.2f})\n"

        return [
            TextContent(
                type="text",
                text=(
                    f"Image edited successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Edit instruction:** {prompt_display}{prompt_suffix}\n"
                    f"{cost_info}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def _expand_image(client: BFLClient, args: dict[str, Any]) -> list[TextContent]:
    """Handle expand_image tool (outpainting)."""
    # Handle image input
    image_input = args["image"]
    if _is_local_file(image_input):
        try:
            image_input = _encode_image_to_base64(image_input)
        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"Error: {e!s}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading image file: {e!s}")]

    # Build payload
    payload: dict[str, Any] = {"image": image_input}

    # Add expansion directions
    for direction in ["top", "bottom", "left", "right"]:
        if direction in args and args[direction] > 0:
            payload[direction] = args[direction]

    # Check at least one direction specified
    if not any(d in payload for d in ["top", "bottom", "left", "right"]):
        return [
            TextContent(
                type="text",
                text="Error: At least one direction (top, bottom, left, right) must be > 0",
            )
        ]

    # Optional params
    if "prompt" in args:
        payload["prompt"] = args["prompt"]
    if "seed" in args:
        payload["seed"] = args["seed"]
    if "safety_tolerance" in args:
        payload["safety_tolerance"] = args["safety_tolerance"]
    if "output_format" in args:
        payload["output_format"] = args["output_format"]

    try:
        submission = await client.submit("v1/flux-pro-1.0-expand", payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        image_url = _extract_image_url(result)
        cost = result.get("cost")

        # Build expansion summary
        expansions = []
        for d in ["top", "bottom", "left", "right"]:
            if d in args and args[d] > 0:
                expansions.append(f"{d}: {args[d]}px")
        expansion_summary = ", ".join(expansions)

        cost_info = ""
        if cost is not None:
            usd = cost * 0.01
            cost_info = f"**Credits used:** {cost} (${usd:.2f})\n"

        return [
            TextContent(
                type="text",
                text=(
                    f"Image expanded successfully!\n\n"
                    f"**Expansion:** {expansion_summary}\n"
                    f"{cost_info}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def _list_finetunes(client: BFLClient) -> list[TextContent]:
    """Handle list_finetunes tool."""
    try:
        result = await client.list_finetunes()

        finetunes = result.get("finetunes", [])
        if not finetunes:
            return [
                TextContent(
                    type="text",
                    text=(
                        "**No finetuned models found.**\n\n"
                        "You haven't created any finetuned models yet."
                    ),
                )
            ]

        lines = ["**Your Finetuned Models**\n"]
        for ft in finetunes:
            name = ft.get("name", "Unknown")
            status = ft.get("status", "Unknown")
            model_id = ft.get("id", "N/A")
            lines.append(f"- **{name}** (ID: `{model_id}`)")
            lines.append(f"  Status: {status}")

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing finetunes: {e!s}")]


async def _create_variation(client: BFLClient, args: dict[str, Any]) -> list[TextContent]:
    """Handle create_variation tool using image_prompt for Redux-style variations."""
    # Map model name to endpoint
    model_map = {
        "flux-dev": "v1/flux-dev",
        "flux-pro-1.1": "v1/flux-pro-1.1",
        "flux-pro-1.1-ultra": "v1/flux-pro-1.1-ultra",
    }

    model = args.get("model", "flux-pro-1.1")
    endpoint = model_map.get(model, "v1/flux-pro-1.1")

    # Handle image input: local file path, URL, or base64
    image_input = args["image"]
    if _is_local_file(image_input):
        try:
            image_input = _encode_image_to_base64(image_input)
        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"Error: {e!s}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading image file: {e!s}")]

    # Build payload - use image_prompt for variation
    payload: dict[str, Any] = {
        "image_prompt": image_input,
    }

    # Add text prompt if provided (for guided variations)
    if "prompt" in args and args["prompt"]:
        payload["prompt"] = args["prompt"]

    # Handle aspect ratio
    if "aspect_ratio" in args:
        payload["aspect_ratio"] = args["aspect_ratio"]

    # Optional params
    if "seed" in args:
        payload["seed"] = args["seed"]
    if "safety_tolerance" in args:
        payload["safety_tolerance"] = args["safety_tolerance"]
    if "output_format" in args:
        payload["output_format"] = args["output_format"]

    try:
        # Submit and wait
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        # Extract image URL and cost
        image_url = _extract_image_url(result)
        cost = result.get("cost")

        prompt_info = ""
        if "prompt" in args and args["prompt"]:
            prompt_display = args["prompt"][:80]
            prompt_suffix = "..." if len(args["prompt"]) > 80 else ""
            prompt_info = f"**Guidance:** {prompt_display}{prompt_suffix}\n"

        cost_info = ""
        if cost is not None:
            usd = cost * 0.01
            cost_info = f"**Credits used:** {cost} (${usd:.2f})\n"

        return [
            TextContent(
                type="text",
                text=(
                    f"Variation created successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"{prompt_info}"
                    f"{cost_info}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


def _is_local_file(image_input: str) -> bool:
    """Check if input is a local file path (not URL or base64)."""
    # URLs are passed directly
    if image_input.startswith(("http://", "https://", "data:")):
        return False
    # Base64 strings are typically very long with no path separators after initial chars
    if len(image_input) > 500 and "/" not in image_input[10:]:
        return False
    # Check if the path exists on the filesystem
    return Path(image_input).exists()


def _encode_image_to_base64(file_path: str) -> str:
    """Read a local image file and return base64-encoded data."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image_bytes = path.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


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
