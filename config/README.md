# Research Assistant Configuration

This directory contains configuration files for the Research Assistant.

## Files

- `settings.yaml.example` - Example YAML configuration file
- `prompts/` - Custom prompt templates

## Usage

Copy `settings.yaml.example` to `settings.yaml` and customize as needed:

```bash
cp settings.yaml.example settings.yaml
```

Then load it in your code:

```python
from src.utils.config import load_config

config = load_config("config/settings.yaml")
```
