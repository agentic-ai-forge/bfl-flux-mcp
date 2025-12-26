# BFL Flux MCP Server

[![Pipeline Status](https://gitlab.com/agentic.ai.forge/bfl-flux-mcp/badges/main/pipeline.svg)](https://gitlab.com/agentic.ai.forge/bfl-flux-mcp/-/pipelines)
[![Coverage](https://gitlab.com/agentic.ai.forge/bfl-flux-mcp/badges/main/coverage.svg)](https://gitlab.com/agentic.ai.forge/bfl-flux-mcp/-/commits/main)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimal Python MCP server for [Black Forest Labs](https://blackforestlabs.ai/) Flux image generation API.

## Features

- **Three focused tools:**
  - `generate_image` - Text-to-image with model selection
  - `edit_image` - Image editing with natural language
  - `check_credits` - Verify API key and credit balance

- **All Flux models supported:**
  - Generation: `flux-pro-1.1` (default), `flux-pro-1.1-ultra`, `flux-2-pro`, `flux-2-flex`, `flux-pro`, `flux-dev`
  - Editing: `kontext-pro`, `kontext-max`, `fill-pro`

- **Full API parameter support:**
  - Aspect ratio or custom dimensions
  - Prompt upsampling for enhanced output
  - Output format (PNG/JPEG)
  - Safety tolerance levels
  - Model-specific: guidance, steps, raw mode

## Installation

```bash
# With uv (recommended)
uv sync

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
      "args": ["run", "--project", "/path/to/bfl-flux-mcp", "bfl-flux-mcp"],
      "env": {
        "BFL_API_KEY": "${BFL_API_KEY}"
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
        "BFL_API_KEY": "${BFL_API_KEY}"
      }
    }
  }
}
```

## Usage

### Check Credits

Verify your API key is working and check balance:

```
Check my BFL credits
```

### Generate Image

```
Generate an image of a mountain landscape at sunset, photorealistic style
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `prompt` | Text description (required) | - |
| `model` | Model to use | `flux-pro-1.1` |
| `aspect_ratio` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9`, `9:21` | `1:1` |
| `width`/`height` | Custom dimensions (multiple of 16, max 2048) | - |
| `seed` | For reproducibility | - |
| `safety_tolerance` | 0 (strict) to 6 (permissive) | 2 |
| `prompt_upsampling` | Enhance prompt for richer output | `false` |
| `output_format` | `png` or `jpeg` | `png` |
| `guidance` | Prompt adherence 1.5-10 (flux-2-flex only) | - |
| `steps` | Generation steps 1-50 (flux-2-flex only) | - |
| `raw` | Raw mode (flux-pro-1.1-ultra only) | `false` |

### Edit Image

```
Edit this image to add a rainbow in the sky
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `prompt` | Edit instructions (required) | - |
| `image` | Base64-encoded image or URL (required) | - |
| `model` | `kontext-pro`, `kontext-max`, `fill-pro` | `kontext-pro` |
| `aspect_ratio` | Output aspect ratio | preserve |
| `seed` | For reproducibility | - |
| `safety_tolerance` | 0-2 | 2 |

## Pricing

| Model | Credits | USD |
|-------|---------|-----|
| flux-pro-1.1 | 4 | $0.04 |
| flux-pro-1.1-ultra | 6 | $0.06 |
| flux-2-pro | 5 | $0.05 |
| flux-2-flex | 1-5 | $0.01-0.05 |
| kontext-pro | 4 | $0.04 |
| kontext-max | 8 | $0.08 |

*1 credit = $0.01 USD*

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=bfl_flux_mcp --cov-report=term-missing
```

## License

MIT - see [LICENSE](LICENSE)

## Links

- [BFL API Documentation](https://docs.bfl.ml)
- [BFL Playground](https://playground.bfl.ai) (free testing)
- [GitLab Repository](https://gitlab.com/agentic.ai.forge/bfl-flux-mcp)
- [GitHub Mirror](https://github.com/agentic-ai-forge/bfl-flux-mcp)
