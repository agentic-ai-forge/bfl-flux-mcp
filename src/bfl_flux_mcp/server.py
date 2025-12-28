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
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Re-export for backwards compatibility with tests
from .client import BFLClient, get_client  # noqa: F401
from .handlers import (
    handle_check_credits,
    handle_create_variation,
    handle_edit_image,
    handle_expand_image,
    handle_generate_image,
    handle_list_finetunes,
    handle_save_image,
)
from .tools import get_tools
from .utils import encode_image_to_base64 as _encode_image_to_base64  # noqa: F401
from .utils import extract_image_url as _extract_image_url  # noqa: F401
from .utils import is_local_file as _is_local_file  # noqa: F401

# Backwards-compatible handler aliases for tests
_check_credits = handle_check_credits
_save_image = handle_save_image
_generate_image = handle_generate_image
_edit_image = handle_edit_image
_expand_image = handle_expand_image
_list_finetunes = handle_list_finetunes
_create_variation = handle_create_variation


# --- MCP Server ---

server = Server("bfl-flux")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool.

    Args:
        name: Tool name to execute
        arguments: Tool arguments

    Returns:
        List of TextContent with results
    """
    # save_image doesn't need BFL client
    if name == "save_image":
        return await _save_image(arguments)

    client = get_client()

    try:
        if name == "generate_image":
            return await handle_generate_image(client, arguments)
        elif name == "edit_image":
            return await handle_edit_image(client, arguments)
        elif name == "check_credits":
            return await handle_check_credits(client)
        elif name == "expand_image":
            return await handle_expand_image(client, arguments)
        elif name == "list_finetunes":
            return await handle_list_finetunes(client)
        elif name == "create_variation":
            return await handle_create_variation(client, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        await client.close()


def main():
    """Run the MCP server."""
    asyncio.run(_run_server())


async def _run_server():
    """Async server runner."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
