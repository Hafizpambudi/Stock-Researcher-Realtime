# Custom Prompt Templates

This directory contains custom prompt templates for the Research Assistant.

## Available Templates

- `research_system.txt` - System prompt for research agent
- `summarization.txt` - Prompt for text summarization
- `analysis.txt` - Prompt for content analysis
- `report_generation.txt` - Prompt for report generation

## Usage

Load prompts in your code:

```python
from pathlib import Path

prompts_dir = Path(__file__).parent

with open(prompts_dir / "research_system.txt") as f:
    system_prompt = f.read()
```
