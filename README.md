# BFL Flux MCP Server

Minimal Python MCP server for Black Forest Labs Flux image generation.

## Features

- **Two focused tools:**
  - `generate_image` - Text-to-image with model selection
  - `edit_image` - Image editing with natural language

- **All Flux models supported:**
  - Generation: `flux-pro`, `flux-dev`, `flux-pro-1.1`, `flux-pro-1.1-ultra`, `flux-2-pro`, `flux-2-dev`
  - Editing: `kontext-pro`, `kontext-max`, `fill-pro`

## Installation

```bash
# With uv (recommended)
uv pip install -e .

# With pip
pip install -e .
```

## Configuration

Get your API key at [api.bfl.ml](https://api.bfl.ml)

```bash
export BFL_API_KEY="your-api-key-here"
```

### Claude Code / MCP Config

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "bfl-flux": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/bfl-flux-mcp", "bfl-flux-mcp"],
      "env": {
        "BFL_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Or with uvx (once published):

```json
{
  "mcpServers": {
    "bfl-flux": {
      "command": "uvx",
      "args": ["bfl-flux-mcp"],
      "env": {
        "BFL_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Usage

### Generate Image

```
Generate an image of a mountain landscape at sunset, photorealistic style
```

Parameters:
- `prompt` (required): Text description
- `model`: `flux-pro-1.1` (default), `flux-pro`, `flux-dev`, `flux-2-pro`, etc.
- `aspect_ratio`: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`
- `width`/`height`: Override aspect ratio (must be multiple of 32)
- `seed`: For reproducibility
- `safety_tolerance`: 0-6 (2 default)

### Edit Image

```
Edit this image to add a rainbow in the sky
```

Parameters:
- `prompt` (required): Edit instructions
- `image` (required): Base64 or URL
- `model`: `kontext-pro` (default), `kontext-max`, `fill-pro`
- `aspect_ratio`: Output ratio
- `seed`: For reproducibility
- `safety_tolerance`: 0-2

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run linter
ruff check .

# Run tests
pytest
```

## License

MIT
