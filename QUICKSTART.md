# Quick Start Guide

Get up and running with the Research Assistant in seconds using uv.

## One-Liner Setup

```bash
# Install uv and setup the project (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh && cd "D:\Portfolio Project" && uv sync && .venv\Scripts\activate
```

```powershell
# Install uv and setup the project (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" && cd "D:\Portfolio Project" && uv sync && .venv\Scripts\activate
```

## Essential Commands

### First Time Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Setup project
cd "D:\Portfolio Project"
uv sync

# 3. Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/macOS
```

### Daily Development

```bash
# Run the research assistant (recommended: use entry point)
uv run research-assistant research "Your topic"

# Alternative: run as module
uv run python -m src.main research "Your topic"

# Start interactive session
uv run research-assistant interactive

# Run tests
uv run pytest tests/ -v

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

### Package Management

```bash
# Install a new package
uv pip install package-name

# Update all dependencies
uv lock --upgrade

# Export dependencies
uv export > requirements.txt
```

## Troubleshooting

### "Command not found: uv"

```bash
# Add uv to PATH (macOS/Linux)
export PATH="$HOME/.local/bin:$PATH"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

```powershell
# Windows - uv should be in PATH automatically
# If not, restart your terminal or add manually
```

### "No Python found"

```bash
# Install Python via uv
uv python install 3.11

# Re-sync
uv sync
```

### "Dependency conflicts"

```bash
# Clean cache and reinstall
uv cache clean
uv sync --reinstall
```

### "Virtual environment issues"

```bash
# Remove and recreate venv
rm -rf .venv  # Unix/macOS
rmdir /s .venv  # Windows

uv sync
```

### "API Key errors"

```bash
# Copy and configure environment
copy .env.example .env  # Windows
cp .env.example .env    # Unix/macOS

# Edit .env and add your API keys
# OPENROUTER_API_KEY=your_key_here
```

## Quick Reference Card

| Goal | Command |
|------|---------|
| **Setup** | `uv sync` |
| **Activate** | `.venv\Scripts\activate` |
| **Run app** | `uv run python -m src.main research "topic"` |
| **Run tests** | `uv run pytest tests/ -v` |
| **Format** | `uv run black src/ tests/` |
| **Lint** | `uv run ruff check src/ tests/` |
| **Install pkg** | `uv pip install package` |
| **Update deps** | `uv lock --upgrade` |
| **Clean cache** | `uv cache clean` |

## Get Help

```bash
# Project documentation
cat README.md
cat UV.md

# uv help
uv --help
uv sync --help
```

## Next Steps

1. Configure your API keys in `.env`
2. Run a test research query
3. Read the full [README.md](README.md) for detailed usage
4. Read [UV.md](UV.md) for comprehensive uv documentation
