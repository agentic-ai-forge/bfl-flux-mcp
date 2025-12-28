"""Tool handler implementations for BFL Flux MCP."""

from pathlib import Path
from typing import Any

import httpx
from mcp.types import TextContent

from .client import BFLClient
from .models import EDIT_MODELS, EXPAND_ENDPOINT, GENERATION_MODELS, VARIATION_MODELS
from .utils import (
    extract_image_url,
    format_cost,
    format_error_response,
    prepare_image_input,
    truncate_prompt,
)


async def handle_check_credits(client: BFLClient) -> list[TextContent]:
    """Check BFL API credit balance.

    Args:
        client: BFL API client

    Returns:
        List containing formatted credit balance
    """
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


async def handle_save_image(args: dict[str, Any]) -> list[TextContent]:
    """Download and save an image to a local file.

    Args:
        args: Tool arguments containing url and path

    Returns:
        List containing success or error message
    """
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
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()

            # Get content type for validation
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Error: URL does not point to an image "
                            f"(content-type: {content_type})"
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
                text=f"Image saved successfully!\n\n**Path:** {path}\n**Size:** {size_kb:.1f} KB",
            )
        ]

    except httpx.HTTPStatusError as e:
        return [TextContent(type="text", text=f"Error downloading image: {e!s}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error saving image: {e!s}")]


async def handle_generate_image(
    client: BFLClient, args: dict[str, Any]
) -> list[TextContent]:
    """Handle generate_image tool.

    Args:
        client: BFL API client
        args: Tool arguments

    Returns:
        List containing generated image URL or error
    """
    model = args.get("model", "flux-pro-1.1")
    endpoint = GENERATION_MODELS.get(model, "v1/flux-pro-1.1")

    # Build payload
    payload: dict[str, Any] = {"prompt": args["prompt"]}

    # Handle dimensions
    if "width" in args and "height" in args:
        payload["width"] = args["width"]
        payload["height"] = args["height"]
    elif "aspect_ratio" in args:
        payload["aspect_ratio"] = args["aspect_ratio"]

    # Optional params
    for param in ["seed", "safety_tolerance", "prompt_upsampling", "output_format"]:
        if param in args:
            payload[param] = args[param]

    # Model-specific params
    if model == "flux-2-flex":
        for param in ["guidance", "steps"]:
            if param in args:
                payload[param] = args[param]
    if model == "flux-pro-1.1-ultra" and args.get("raw"):
        payload["raw"] = True

    try:
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        image_url = extract_image_url(result)
        cost = result.get("cost")

        prompt_display, suffix = truncate_prompt(args["prompt"])

        return [
            TextContent(
                type="text",
                text=(
                    f"Image generated successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Prompt:** {prompt_display}{suffix}\n"
                    f"{format_cost(cost)}"
                    f"**Image URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [format_error_response(str(e))]


async def handle_edit_image(
    client: BFLClient, args: dict[str, Any]
) -> list[TextContent]:
    """Handle edit_image tool.

    Args:
        client: BFL API client
        args: Tool arguments

    Returns:
        List containing edited image URL or error
    """
    model = args.get("model", "kontext-pro")
    endpoint = EDIT_MODELS.get(model, "v1/flux-kontext-pro")

    # Handle image input
    image_input = prepare_image_input(args["image"])
    if isinstance(image_input, TextContent):
        return [image_input]

    # Build payload
    payload: dict[str, Any] = {
        "prompt": args["prompt"],
        "input_image": image_input,
    }

    # Optional params
    for param in ["aspect_ratio", "seed", "safety_tolerance"]:
        if param in args:
            payload[param] = args[param]

    try:
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        image_url = extract_image_url(result)
        cost = result.get("cost")

        prompt_display, suffix = truncate_prompt(args["prompt"])

        return [
            TextContent(
                type="text",
                text=(
                    f"Image edited successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"**Edit instruction:** {prompt_display}{suffix}\n"
                    f"{format_cost(cost)}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [format_error_response(str(e))]


async def handle_expand_image(
    client: BFLClient, args: dict[str, Any]
) -> list[TextContent]:
    """Handle expand_image tool (outpainting).

    Args:
        client: BFL API client
        args: Tool arguments

    Returns:
        List containing expanded image URL or error
    """
    # Handle image input
    image_input = prepare_image_input(args["image"])
    if isinstance(image_input, TextContent):
        return [image_input]

    # Build payload
    payload: dict[str, Any] = {"image": image_input}

    # Add expansion directions
    directions = ["top", "bottom", "left", "right"]
    for direction in directions:
        if direction in args and args[direction] > 0:
            payload[direction] = args[direction]

    # Check at least one direction specified
    if not any(d in payload for d in directions):
        return [
            TextContent(
                type="text",
                text="Error: At least one direction (top, bottom, left, right) must be > 0",
            )
        ]

    # Optional params
    for param in ["prompt", "seed", "safety_tolerance", "output_format"]:
        if param in args:
            payload[param] = args[param]

    try:
        submission = await client.submit(EXPAND_ENDPOINT, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        image_url = extract_image_url(result)
        cost = result.get("cost")

        # Build expansion summary
        expansions = [f"{d}: {args[d]}px" for d in directions if d in args and args[d] > 0]
        expansion_summary = ", ".join(expansions)

        return [
            TextContent(
                type="text",
                text=(
                    f"Image expanded successfully!\n\n"
                    f"**Expansion:** {expansion_summary}\n"
                    f"{format_cost(cost)}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [format_error_response(str(e))]


async def handle_list_finetunes(client: BFLClient) -> list[TextContent]:
    """Handle list_finetunes tool.

    Args:
        client: BFL API client

    Returns:
        List containing finetunes information or error
    """
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


async def handle_create_variation(
    client: BFLClient, args: dict[str, Any]
) -> list[TextContent]:
    """Handle create_variation tool using image_prompt for Redux-style variations.

    Args:
        client: BFL API client
        args: Tool arguments

    Returns:
        List containing variation image URL or error
    """
    model = args.get("model", "flux-pro-1.1")
    endpoint = VARIATION_MODELS.get(model, "v1/flux-pro-1.1")

    # Handle image input
    image_input = prepare_image_input(args["image"])
    if isinstance(image_input, TextContent):
        return [image_input]

    # Build payload - use image_prompt for variation
    payload: dict[str, Any] = {"image_prompt": image_input}

    # Add text prompt if provided (for guided variations)
    if "prompt" in args and args["prompt"]:
        payload["prompt"] = args["prompt"]

    # Optional params
    for param in ["aspect_ratio", "seed", "safety_tolerance", "output_format"]:
        if param in args:
            payload[param] = args[param]

    try:
        submission = await client.submit(endpoint, payload)
        task_id = submission["id"]
        polling_url = submission.get("polling_url")

        result = await client.wait_for_completion(task_id, polling_url)

        image_url = extract_image_url(result)
        cost = result.get("cost")

        prompt_info = ""
        if "prompt" in args and args["prompt"]:
            prompt_display, suffix = truncate_prompt(args["prompt"], 80)
            prompt_info = f"**Guidance:** {prompt_display}{suffix}\n"

        return [
            TextContent(
                type="text",
                text=(
                    f"Variation created successfully!\n\n"
                    f"**Model:** {model}\n"
                    f"{prompt_info}"
                    f"{format_cost(cost)}"
                    f"**Result URL:** {image_url}\n\n"
                    f"Note: URL is valid for 10 minutes. Download or use immediately."
                ),
            )
        ]

    except Exception as e:
        return [format_error_response(str(e))]
