# Intelligent Research Assistant

A powerful LangChain-based research assistant that helps users research topics by searching for information, analyzing and summarizing findings, and generating structured reports with citations.

**Now supporting OpenRouter API** - Access multiple LLM providers (OpenAI, Anthropic, Google, and more) through a single unified API.

## Features

- 🔍 **Web Search**: Search for information across multiple sources using DuckDuckGo
- 🌐 **Web Browsing**: Navigate to URLs, extract main content, and discover links with intelligent caching
- 📝 **Summarization**: Condense long articles and documents into concise summaries
- 📚 **Citation Management**: Track sources and generate bibliographies in APA, MLA, and Chicago styles
- 🤖 **Intelligent Agent**: AI-powered research coordination using LangChain agents
- 🔀 **Chain Architecture**: Modular chain system (Sequential, Transform, Router) for flexible workflows
- 📊 **Report Generation**: Generate structured research reports in markdown format
- 🌐 **Multi-Provider Support**: Use OpenRouter to access models from OpenAI, Anthropic, Google, Meta, and more

## Project Structure

```
D:\Portfolio Project\
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   └── research_agent.py   # Main research agent
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── sequential_chain.py # Sequential processing chain
│   │   ├── transform_chain.py  # Data transformation chain
│   │   └── router_chain.py     # Dynamic routing chain
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── browser_tool.py     # Web browsing tool
│   │   ├── search_tool.py      # Web search tool
│   │   ├── summarize_tool.py   # Text summarization tool
│   │   └── cite_tool.py        # Citation management tool
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging configuration
│       ├── config.py           # Configuration management
│       └── helpers.py          # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_utils.py
│   ├── test_tools.py
│   ├── test_chains.py
│   └── test_agents.py
├── examples/
│   └── usage_examples.py       # Example usage scripts
├── data/
│   └── reports/                # Generated reports
├── config/                     # Configuration files
├── .env.example                # Environment variables template
├── .gitignore
├── pyproject.toml              # Package configuration
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- **uv** package manager (recommended) or **pip**
- OpenRouter API key (recommended) or OpenAI API key

### Quick Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package manager written in Rust. It's a drop-in replacement for pip and venv.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Clone/navigate to the project
cd "D:\Portfolio Project"

# 3. Create a virtual environment and install dependencies
uv sync

# 4. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate
```

### Alternative: Install with pip

If you prefer using pip:

```bash
# 1. Navigate to the project
cd "D:\Portfolio Project"

# 2. Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e ".[dev]"
```

### OpenRouter Setup (Recommended)

OpenRouter provides access to multiple LLM providers through a single API. This is the recommended way to use the Research Assistant.

1. **Get an OpenRouter API Key:**
   - Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
   - Sign up or log in to your account
   - Create a new API key

2. **Choose a Model:**
   - OpenRouter offers models from multiple providers:
     - **OpenAI**: `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
     - **Anthropic**: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`
     - **Google**: `google/gemini-pro`, `google/gemini-pro-vision`
     - **Meta**: `meta-llama/llama-3-70b-instruct`
     - And many more!
   - Browse all available models at [https://openrouter.ai/models](https://openrouter.ai/models)

3. **Configure Environment Variables:**
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_MODEL=openai/gpt-4-turbo
   ```

### OpenRouter Reasoning Feature

OpenRouter supports a **reasoning feature** for certain models that provides step-by-step reasoning traces before the final answer. This is useful for understanding the model's thought process and improving transparency.

**Supported Models:**
- **MiniMax**: `minimax/minimax-m2.5`
- **Anthropic Claude**: Some Claude models (check OpenRouter for current support)
- Other reasoning-capable models as they become available

**How to Enable:**

1. **Via Environment Variable:**
   ```env
   OPENROUTER_REASONING_ENABLED=true
   OPENROUTER_MODEL=minimax/minimax-m2.5
   ```

2. **Via Code:**
   ```python
   from src.agents.research_agent import ResearchAgent

   # Enable reasoning when creating the agent
   agent = ResearchAgent.with_openrouter(
       model="minimax/minimax-m2.5",
       reasoning_enabled=True
   )
   ```

**Accessing Reasoning Traces:**

When reasoning is enabled, the model's response may include reasoning details in the response metadata. You can access these via callbacks or by inspecting the raw response:

```python
from langchain_core.callbacks import BaseCallbackHandler

class ReasoningCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        # Access reasoning from response metadata
        if hasattr(response, 'generators') and response.generators:
            for gen in response.generators:
                if hasattr(gen, 'reasoning'):
                    print(f"Reasoning: {gen.reasoning}")
```

**Note:** The reasoning feature is opt-in and disabled by default. It may increase response time and token usage.

### Environment Setup

1. **Copy the example environment file:**

```bash
copy .env.example .env  # Windows
cp .env.example .env    # Unix/macOS
```

2. **Edit `.env` and add your API keys:**

#### Using OpenRouter (Recommended)

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4-turbo

# Optional: Direct OpenAI API (if not using OpenRouter)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Optional: Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Search configuration
SEARCH_ENGINE=duckduckgo
MAX_SEARCH_RESULTS=5
```

#### Using Direct OpenAI API

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Optional: Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Search configuration
SEARCH_ENGINE=duckduckgo
MAX_SEARCH_RESULTS=5
```

## Usage

### Command Line Interface

#### Research a Topic

```bash
# Basic research
python -m src.main research "Latest developments in quantum computing"

# Save report to file
python -m src.main research "Climate change solutions" -o report.md

# Use specific model
python -m src.main research "AI in healthcare" --model gpt-4

# Verbose output
python -m src.main research "Machine learning trends" -v
```

#### Interactive Mode

```bash
python -m src.main interactive
```

In interactive mode, you can:
- Enter research queries
- Type `report` to generate a full report
- Type `reset` to start a new session
- Type `quit` or `exit` to end

### Python API

#### Basic Usage with OpenRouter

```python
from src.agents.research_agent import ResearchAgent

# Create agent using OpenRouter (recommended)
agent = ResearchAgent.with_openrouter()

# Or specify a model
agent = ResearchAgent.with_openrouter(model="anthropic/claude-3-opus")

# Start research session
agent.start_session("Benefits of renewable energy")

# Conduct research
report = agent.research("What are the main benefits of renewable energy?")

# Display report
print(report.to_markdown())

# Save report
with open("report.md", "w") as f:
    f.write(report.to_markdown())
```

#### Basic Usage with Direct OpenAI

```python
from src.agents.research_agent import ResearchAgent

# Create agent using direct OpenAI API
agent = ResearchAgent.with_openai()

# Or specify a model
agent = ResearchAgent.with_openai(model="gpt-4-turbo-preview")

# Start research session
agent.start_session("Benefits of renewable energy")

# Conduct research
report = agent.research("What are the main benefits of renewable energy?")

# Display report
print(report.to_markdown())
```

#### Using the LLM Factory Functions

```python
from src.utils.config import get_openrouter_llm, get_openai_llm
from src.agents.research_agent import ResearchAgent

# Get OpenRouter-configured LLM
llm = get_openrouter_llm(model="google/gemini-pro")

# Get direct OpenAI LLM
llm = get_openai_llm(model="gpt-4-turbo-preview")

# Create agent with custom LLM
agent = ResearchAgent(llm=llm)
```

#### Using Tools Directly

```python
from src.tools.search_tool import SearchTool
from src.tools.summarize_tool import SummarizeTool
from src.tools.cite_tool import CiteTool
from src.tools.browser_tool import BrowserTool

# Search
search_tool = SearchTool()
results = search_tool.run({"query": "latest AI developments"})

# Browse Web Pages
browser_tool = BrowserTool()

# Visit a URL and extract content
result = browser_tool.visit("https://example.com")
print(f"Title: {result.title}")
print(f"Content: {result.content[:500]}")

# Extract main content (removes ads, navigation, etc.)
content_result = browser_tool.extract_content("https://example.com/article")
print(f"Clean content: {content_result.content}")

# Get all links from a page
links_result = browser_tool.get_links("https://example.com")
print(f"Found {len(links_result.links)} links")

# Search the web
search_results = browser_tool.search("Python programming")
print(search_results)

# Browse with agent
agent = ResearchAgent()
page_info = agent.browse_url("https://example.com")
print(f"Page title: {page_info['title']}")

# Summarize
summarize_tool = SummarizeTool(llm=llm)
summary = summarize_tool.run({"text": long_article_content})

# Cite
cite_tool = CiteTool()
citation = cite_tool.run({
    "content": "https://example.com/article",
    "citation_type": "web",
    "metadata": {"title": "Article Title", "author": "Author Name"}
})

# Get bibliography
bibliography = cite_tool.get_bibliography("apa")
```

#### Using Chains

```python
from src.chains.sequential_chain import ResearchSequentialChain
from src.chains.transform_chain import ResearchTransformChain
from src.chains.router_chain import ResearchRouterChain

# Sequential Chain - Multi-step processing
sequential = ResearchSequentialChain.create_research_chain(
    llm=llm,
    include_search=True,
    include_analysis=True,
    include_summary=True,
)
result = sequential.invoke({"query": "Research topic"})

# Transform Chain - Format transformation
transform = ResearchTransformChain.create_format_chain(
    output_format="markdown",
    llm=llm,
)
formatted = transform.invoke({"content": raw_content})

# Router Chain - Dynamic routing
router = ResearchRouterChain.create_research_router(
    llm=llm,
    search_chain=search_chain,
    summarize_chain=summarize_chain,
    analyze_chain=analyze_chain,
)
result = router.invoke({"query": "Summarize this article"})
```

#### Complete Workflow Example

```python
from src.agents.research_agent import ResearchAgent

# Setup with OpenRouter
agent = ResearchAgent.with_openrouter(model="openai/gpt-4-turbo")
agent.setup_chains()  # Enable advanced chain features

# Multi-query research
topic = "Impact of climate change on biodiversity"
agent.start_session(topic)

queries = [
    "How does climate change affect species extinction rates?",
    "What are the impacts of rising temperatures on ecosystems?",
    "Which species are most vulnerable to climate change?",
]

for query in queries:
    agent.research(query, generate_report=False)

# Generate final report
report = agent.generate_report()
print(report.to_markdown())

# Session summary
summary = agent.get_session_summary()
print(f"Queries: {summary['queries_count']}")
print(f"Findings: {summary['findings_count']}")
print(f"Citations: {summary['citations_count']}")
```

## Architecture

### Agents

The `ResearchAgent` is the main orchestrator that:
- Manages research sessions
- Coordinates tools and chains
- Generates structured reports
- Tracks citations and findings

### Chains

Three chain types provide flexible processing:

1. **Sequential Chain**: Processes input through multiple steps in order
2. **Transform Chain**: Applies transformations (formatting, filtering, enriching)
3. **Router Chain**: Dynamically routes queries to appropriate handlers

### Tools

Four core tools enable research capabilities:

1. **BrowserTool**: Web browsing with content extraction, link discovery, and caching
   - `visit(url)`: Navigate to a URL and extract content
   - `extract_content(url)`: Extract main content (removes ads, navigation)
   - `get_links(url)`: Get all links from a page
   - `search(query)`: Search the web
   - `scroll(url)`: Simulate scrolling for dynamic content
   - Features: LRU caching, rate limiting, proper User-Agent headers

2. **SearchTool**: Web search using DuckDuckGo

3. **SummarizeTool**: LLM-based text summarization

4. **CiteTool**: Citation creation and bibliography generation

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required (or OPENAI_API_KEY) |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Model to use via OpenRouter | `openai/gpt-4-turbo` |
| `OPENROUTER_REASONING_ENABLED` | Enable reasoning traces for supported models | `false` |
| `OPENAI_API_KEY` | Direct OpenAI API key | Optional |
| `OPENAI_MODEL` | Direct OpenAI model | `gpt-4-turbo-preview` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `SEARCH_ENGINE` | Search engine to use | `duckduckgo` |
| `MAX_SEARCH_RESULTS` | Max search results | `5` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEFAULT_REPORT_FORMAT` | Report format | `markdown` |

### Custom Configuration

Create a `config/settings.yaml` file for additional configuration:

```yaml
research:
  max_iterations: 10
  timeout_seconds: 300
  
report:
  include_executive_summary: true
  include_key_findings: true
  max_findings: 5
```

## Testing

Run the test suite:

```bash
# With uv (recommended)
uv run pytest tests/ -v

# With uv and coverage
uv run pytest tests/ -v --cov=src --cov-report=html

# With pip
pytest tests/ -v

# With pip and coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tools.py -v
```

## Development Workflow

### Using uv (Recommended)

```bash
# Sync environment and install dependencies
uv sync

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate

# Run the application
uv run python -m src.main research "Your topic"

# Run tests
uv run pytest tests/ -v

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Install additional optional dependencies
uv pip install anthropic>=0.8.0

# Update all dependencies
uv lock --upgrade

# Export dependencies for pip users
uv export > requirements.txt
```

### Using pip (Alternative)

```bash
# Activate virtual environment
source venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the application
python -m src.main research "Your topic"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Code Quality Commands

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Mypy** for type checking

```bash
# Format all code
uv run black src/ tests/

# Lint all code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Examples

See `examples/usage_examples.py` for comprehensive examples:

```bash
python examples/usage_examples.py
```

Examples include:
- Basic research workflow
- Direct chain usage
- Direct tool usage
- Complete research workflow
- Mock data demonstration

## Citation Styles

The CiteTool supports multiple citation formats:

- **APA**: American Psychological Association
- **MLA**: Modern Language Association
- **Chicago**: Chicago Manual of Style

```python
# Get bibliography in different styles
apa_bib = cite_tool.get_bibliography("apa")
mla_bib = cite_tool.get_bibliography("mla")
chicago_bib = cite_tool.get_bibliography("chicago")
```

## Troubleshooting

### Missing API Key

```
Error: Missing required environment variables: ['OPENROUTER_API_KEY or OPENAI_API_KEY']
```

**Solution**: Copy `.env.example` to `.env` and add your API key.

#### For OpenRouter:
1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Create an API key
3. Add it to your `.env` file:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

#### For Direct OpenAI:
1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create an API key
3. Add it to your `.env` file:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

### Search Not Working

```
Warning: DuckDuckGo search not available, using mock search
```

**Solution**: Install the duckduckgo-search package:

```bash
pip install duckduckgo-search
```

### Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution**: Ensure you're running from the project root or add it to PYTHONPATH:

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;D:\Portfolio Project

# Unix/macOS
export PYTHONPATH=$PYTHONPATH:/path/to/Portfolio\ Project
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Setting Up for Development

```bash
# Clone your fork
git clone https://github.com/your-username/research-assistant.git
cd research-assistant

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Sync environment and install with dev dependencies
uv sync

# Activate the environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/macOS
```

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Mypy** for type checking

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [OpenAI](https://openai.com/) - LLM provider
- [DuckDuckGo](https://duckduckgo.com/) - Search engine
