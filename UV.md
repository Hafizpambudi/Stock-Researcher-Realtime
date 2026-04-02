# UV Package Manager Guide

This document explains how to use [uv](https://github.com/astral-sh/uv) with the Research Assistant project.

## What is uv?

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package manager written in Rust. It's designed as a drop-in replacement for pip, pip-tools, and virtualenv.

### Key Benefits

- **10-100x faster** than pip for package installation
- **Single binary** - no Python required to run uv itself
- **Pip-compatible** - uses the same dependency resolver and package index
- **Built-in virtual environment** management
- **Lock file support** for reproducible builds
- **Django/Rust-style** project management with `pyproject.toml`

## Installation

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verify Installation

```bash
uv --version
```

### Upgrade uv

```bash
uv self update
```

## Project Setup

### Initial Setup

```bash
# Navigate to project directory
cd "D:\Portfolio Project"

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate
```

### What `uv sync` Does

1. Creates a `.venv` virtual environment (if it doesn't exist)
2. Reads dependencies from `pyproject.toml`
3. Installs all dependencies (including dev dependencies by default)
4. Creates/updates `uv.lock` for reproducible builds

## Common Commands

### Installing Dependencies

```bash
# Sync all dependencies (including dev)
uv sync

# Sync production dependencies only
uv sync --no-dev

# Install a specific package
uv pip install package-name

# Install in editable mode
uv pip install -e .

# Install with optional dependencies
uv pip install ".[dev]"
uv pip install ".[vectorstore]"
uv pip install ".[all]"
```

### Running Commands

```bash
# Run a command in the project's virtual environment
uv run python -m src.main research "Your topic"
uv run pytest tests/ -v
uv run black src/
uv run ruff check src/
uv run mypy src/
```

### Managing Dependencies

```bash
# Add a new dependency
uv pip install requests

# Add a dev dependency
uv pip install --dev pytest

# Remove a dependency
uv pip uninstall package-name

# Update all dependencies
uv lock --upgrade

# Update a specific dependency
uv lock --upgrade package-name

# Show dependency tree
uv pip tree

# List installed packages
uv pip list
```

### Lock File Management

```bash
# Generate/update lock file
uv lock

# Export lock file to requirements.txt
uv export > requirements.txt

# Export dev dependencies too
uv export --dev > requirements-dev.txt

# Export with hashes for security
uv export --hashes > requirements.txt
```

### Virtual Environment Management

```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.11

# Create in custom location
uv venv /path/to/venv

# Remove virtual environment
rm -rf .venv  # Unix/macOS
rmdir /s .venv  # Windows
```

### Python Version Management

uv can automatically manage Python versions:

```bash
# Install a specific Python version
uv python install 3.11

# List available Python versions
uv python list

# Set project Python version (creates .python-version)
uv python pin 3.11
```

## Project-Specific Workflows

### Development Workflow

```bash
# 1. Clone and setup
git clone https://github.com/your-username/research-assistant.git
cd research-assistant
uv sync

# 2. Make changes to code
# ... edit files ...

# 3. Run tests
uv run pytest tests/ -v

# 4. Format and lint
uv run black src/ tests/
uv run ruff check src/ tests/

# 5. Type check
uv run mypy src/
```

### Running the Application

```bash
# Research a topic
uv run python -m src.main research "Latest developments in AI"

# Interactive mode
uv run python -m src.main interactive

# With custom model
uv run python -m src.main research "Quantum computing" --model anthropic/claude-3-opus
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_tools.py -v

# Run with verbose output
uv run pytest tests/ -vv
```

## Migration from pip/venv

### If You Have an Existing venv

```bash
# 1. Remove old virtual environment
rm -rf venv  # or venv\ on Windows

# 2. Remove old pip files
rm -f requirements.lock pip-lock.txt

# 3. Setup with uv
uv sync

# 4. Verify installation
uv pip list
```

### Converting requirements.txt

If you have a complex `requirements.txt`:

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Or migrate dependencies to pyproject.toml manually
# Then run:
uv sync
```

### Compatibility Notes

- uv is **pip-compatible** - all pip packages work with uv
- uv uses the same PyPI index as pip
- uv supports the same dependency specifiers (`>=`, `~=`, `[]`, etc.)
- uv respects `.python-version` files (like pyenv)

## Troubleshooting

### Common Issues

#### "No Python installation found"

```bash
# Install Python via uv
uv python install 3.11

# Or specify Python path
uv venv --python /usr/bin/python3.11
```

#### "Dependency conflict"

```bash
# Clear cache and retry
uv cache clean
uv sync

# Or upgrade conflicting package
uv lock --upgrade package-name
```

#### "Lock file out of date"

```bash
# Regenerate lock file
uv lock --upgrade
```

#### "Package not found"

```bash
# Clear cache
uv cache clean

# Try installing directly
uv pip install package-name --reinstall
```

### Getting Help

```bash
# uv help
uv --help

# Specific command help
uv sync --help
uv pip install --help

# Check uv version
uv --version

# Show uv configuration
uv config --help
```

## Configuration

### Project Configuration (pyproject.toml)

```toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "black>=23.0.0",
]

[tool.uv.sources]
# Custom package sources if needed
```

### Global Configuration (~/.config/uv/uv.toml)

```toml
# Global uv configuration
index-url = "https://pypi.org/simple"
```

### Environment Variables

```bash
# Custom index URL
export UV_INDEX_URL="https://pypi.org/simple"

# Offline mode
export UV_OFFLINE="true"

# Custom cache directory
export UV_CACHE_DIR="/path/to/cache"
```

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [uv PyPI Package](https://pypi.org/project/uv/)
- [Astral Blog](https://astral.sh/blog)

## Quick Reference

| Task | Command |
|------|---------|
| Setup project | `uv sync` |
| Run command | `uv run <command>` |
| Install package | `uv pip install <package>` |
| Update deps | `uv lock --upgrade` |
| Export deps | `uv export > requirements.txt` |
| List packages | `uv pip list` |
| Show tree | `uv pip tree` |
| Clean cache | `uv cache clean` |
| Python version | `uv python install 3.11` |
