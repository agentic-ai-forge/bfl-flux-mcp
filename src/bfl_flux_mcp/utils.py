"""Utility functions for image handling and response formatting."""

import base64
from pathlib import Path
from typing import Any

from mcp.types import TextContent


def is_local_file(image_input: str) -> bool:
    """Check if input is a local file path (not URL or base64).

    Args:
        image_input: String that could be a file path, URL, or base64 data

    Returns:
        True if input is an existing local file path
    """
    # URLs are passed directly
    if image_input.startswith(("http://", "https://", "data:")):
        return False
    # Base64 strings are typically very long with no path separators after initial chars
    if len(image_input) > 500 and "/" not in image_input[10:]:
        return False
    # Check if the path exists on the filesystem
    return Path(image_input).exists()


def encode_image_to_base64(file_path: str) -> str:
    """Read a local image file and return base64-encoded data.

    Args:
        file_path: Path to the image file

    Returns:
        Base64-encoded string of the image data

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image_bytes = path.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_image_url(result: dict[str, Any]) -> str:
    """Extract image URL from API result.

    Args:
        result: API response containing result data

    Returns:
        Image URL string

    Raises:
        ValueError: If no image URL found in result
    """
    result_data = result.get("result", {})

    if isinstance(result_data, dict) and "sample" in result_data:
        return result_data["sample"]
    elif isinstance(result_data, list) and len(result_data) > 0:
        item = result_data[0]
        return item if isinstance(item, str) else item.get("sample", "")

    raise ValueError("No image URL found in result")


def format_cost(cost: float | None) -> str:
    """Format cost information for display.

    Args:
        cost: Cost in credits (1 credit = $0.01)

    Returns:
        Formatted cost string or empty string if no cost
    """
    if cost is None:
        return ""
    usd = cost * 0.01
    return f"**Credits used:** {cost} (${usd:.2f})\n"


def truncate_prompt(prompt: str, max_length: int = 100) -> tuple[str, str]:
    """Truncate a prompt for display.

    Args:
        prompt: The prompt text
        max_length: Maximum length before truncation

    Returns:
        Tuple of (truncated_prompt, suffix) where suffix is "..." if truncated
    """
    if len(prompt) > max_length:
        return prompt[:max_length], "..."
    return prompt, ""


def format_success_response(
    title: str,
    model: str | None = None,
    prompt: str | None = None,
    cost: float | None = None,
    url: str | None = None,
    extra_info: str = "",
) -> TextContent:
    """Format a standard success response.

    Args:
        title: Response title (e.g., "Image generated successfully!")
        model: Model name used
        prompt: Prompt or instruction used
        cost: Cost in credits
        url: Result URL
        extra_info: Additional information to include

    Returns:
        Formatted TextContent response
    """
    lines = [f"{title}\n"]

    if model:
        lines.append(f"**Model:** {model}")

    if prompt:
        prompt_display, suffix = truncate_prompt(prompt)
        lines.append(f"**Prompt:** {prompt_display}{suffix}")

    if extra_info:
        lines.append(extra_info)

    if cost is not None:
        lines.append(format_cost(cost).rstrip())

    if url:
        lines.append(f"**Image URL:** {url}")
        lines.append("")
        lines.append("Note: URL is valid for 10 minutes. Download or use immediately.")

    return TextContent(type="text", text="\n".join(lines))


def format_error_response(message: str) -> TextContent:
    """Format an error response.

    Args:
        message: Error message

    Returns:
        Formatted TextContent with error
    """
    return TextContent(type="text", text=f"Error: {message}")


def prepare_image_input(image_input: str) -> str | TextContent:
    """Prepare image input for API, converting local files to base64.

    Args:
        image_input: Image path, URL, or base64 string

    Returns:
        Processed image string ready for API, or TextContent error
    """
    if is_local_file(image_input):
        try:
            return encode_image_to_base64(image_input)
        except FileNotFoundError as e:
            return format_error_response(str(e))
        except Exception as e:
            return format_error_response(f"Error reading image file: {e!s}")
    return image_input
