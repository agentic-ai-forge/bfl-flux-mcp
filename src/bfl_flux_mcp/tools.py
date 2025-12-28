"""MCP Tool definitions for BFL Flux API."""

from mcp.types import Tool

# Common schema fragments
ASPECT_RATIO_SCHEMA = {
    "type": "string",
    "description": "Aspect ratio (default: 1:1)",
    "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
    "default": "1:1",
}

SEED_SCHEMA = {
    "type": "integer",
    "description": "Seed for reproducibility (optional)",
}

SAFETY_TOLERANCE_SCHEMA = {
    "type": "integer",
    "description": "Safety filter (0=strict, 6=permissive, default: 2)",
    "minimum": 0,
    "maximum": 6,
    "default": 2,
}

OUTPUT_FORMAT_SCHEMA = {
    "type": "string",
    "description": "Output format (default: png)",
    "enum": ["png", "jpeg"],
    "default": "png",
}

IMAGE_INPUT_SCHEMA = {
    "type": "string",
    "description": "Image: local file path, URL, or base64-encoded data",
}


def get_tools() -> list[Tool]:
    """Get all available MCP tools.

    Returns:
        List of Tool definitions
    """
    return [
        _generate_image_tool(),
        _check_credits_tool(),
        _edit_image_tool(),
        _save_image_tool(),
        _expand_image_tool(),
        _list_finetunes_tool(),
        _create_variation_tool(),
    ]


def _generate_image_tool() -> Tool:
    """Tool definition for generate_image."""
    return Tool(
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
                "aspect_ratio": ASPECT_RATIO_SCHEMA,
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
                "seed": SEED_SCHEMA,
                "safety_tolerance": SAFETY_TOLERANCE_SCHEMA,
                "prompt_upsampling": {
                    "type": "boolean",
                    "description": "Enhance prompt for richer output (recommended for logos)",
                    "default": False,
                },
                "output_format": OUTPUT_FORMAT_SCHEMA,
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
    )


def _check_credits_tool() -> Tool:
    """Tool definition for check_credits."""
    return Tool(
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
    )


def _edit_image_tool() -> Tool:
    """Tool definition for edit_image."""
    return Tool(
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
                    "description": ("Image to edit: local file path, URL, or base64-encoded data"),
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
                "seed": SEED_SCHEMA,
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
    )


def _save_image_tool() -> Tool:
    """Tool definition for save_image."""
    return Tool(
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
    )


def _expand_image_tool() -> Tool:
    """Tool definition for expand_image."""
    return Tool(
        name="expand_image",
        description=(
            "Expand an image by adding pixels to any side (outpainting). "
            "Specify pixels to add to top, bottom, left, and/or right. "
            "The AI will generate content that seamlessly extends the image."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image": IMAGE_INPUT_SCHEMA,
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
                "seed": SEED_SCHEMA,
                "safety_tolerance": SAFETY_TOLERANCE_SCHEMA,
                "output_format": OUTPUT_FORMAT_SCHEMA,
            },
            "required": ["image"],
        },
    )


def _list_finetunes_tool() -> Tool:
    """Tool definition for list_finetunes."""
    return Tool(
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
    )


def _create_variation_tool() -> Tool:
    """Tool definition for create_variation."""
    return Tool(
        name="create_variation",
        description=(
            "Create variations of an existing image using Redux. "
            "Generates images that maintain the essence of the original "
            "while applying optional text-guided modifications."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image": IMAGE_INPUT_SCHEMA,
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
                "aspect_ratio": ASPECT_RATIO_SCHEMA,
                "seed": SEED_SCHEMA,
                "safety_tolerance": SAFETY_TOLERANCE_SCHEMA,
                "output_format": OUTPUT_FORMAT_SCHEMA,
            },
            "required": ["image"],
        },
    )
