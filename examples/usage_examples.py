"""
Example usage script for the Research Assistant.

This script demonstrates how to use the Research Assistant
for various research tasks and showcases its capabilities.

Supports both OpenRouter API (recommended) and direct OpenAI API.
"""

import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
os.sys.path.insert(0, str(project_root))

from src.agents.research_agent import ResearchAgent
from src.chains.router_chain import ResearchRouterChain
from src.chains.sequential_chain import ResearchSequentialChain
from src.chains.transform_chain import ResearchTransformChain
from src.tools.cite_tool import CiteTool
from src.tools.search_tool import SearchTool
from src.tools.summarize_tool import SummarizeTool
from src.utils.config import get_openrouter_llm, get_openai_llm, get_settings
from src.utils.logger import setup_logger


def example_basic_research_openrouter() -> None:
    """Demonstrate basic research using OpenRouter API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Research with OpenRouter")
    print("=" * 60)

    # Create agent using OpenRouter (recommended)
    agent = ResearchAgent.with_openrouter(temperature=0.7)

    # Start research session
    agent.start_session("Benefits of renewable energy")

    # Conduct research
    print("\n🔍 Researching: Benefits of renewable energy\n")

    # Use the agent to research
    report = agent.research("What are the main benefits of renewable energy?")

    # Display results
    print("\n📊 Research Report:\n")
    print(report.to_markdown())

    # Session summary
    summary = agent.get_session_summary()
    print(f"\n📈 Session Summary: {summary}")


def example_basic_research_openai() -> None:
    """Demonstrate basic research using direct OpenAI API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1b: Basic Research with Direct OpenAI")
    print("=" * 60)

    # Create agent using direct OpenAI API
    agent = ResearchAgent.with_openai(temperature=0.7)

    # Start research session
    agent.start_session("Benefits of renewable energy")

    # Conduct research
    print("\n🔍 Researching: Benefits of renewable energy\n")

    # Use the agent to research
    report = agent.research("What are the main benefits of renewable energy?")

    # Display results
    print("\n📊 Research Report:\n")
    print(report.to_markdown())

    # Session summary
    summary = agent.get_session_summary()
    print(f"\n📈 Session Summary: {summary}")


def example_chain_usage() -> None:
    """Demonstrate using chains directly."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using Chains Directly")
    print("=" * 60)

    # Get OpenRouter-configured LLM
    llm = get_openrouter_llm(model="openai/gpt-4-turbo", temperature=0.7)

    # Sequential Chain Example
    print("\n📋 Sequential Chain Demo\n")

    sequential_chain = ResearchSequentialChain.create_research_chain(
        llm=llm,
        include_search=True,
        include_analysis=True,
        include_summary=True,
    )

    result = sequential_chain.invoke({"query": "Machine learning applications in healthcare"})
    print(f"Sequential Chain Result:\n{result.get('result', 'N/A')[:500]}...\n")

    # Transform Chain Example
    print("\n🔄 Transform Chain Demo\n")

    transform_chain = ResearchTransformChain.create_format_chain(
        output_format="markdown", llm=llm
    )

    sample_content = """
    Machine learning is revolutionizing healthcare by enabling
    predictive diagnostics, personalized treatment plans, and
    automated medical imaging analysis. Key applications include
    disease prediction, drug discovery, and patient monitoring.
    """

    result = transform_chain.invoke({"content": sample_content})
    print(f"Transform Chain Result:\n{result.get('transformed', 'N/A')}\n")

    # Router Chain Example
    print("\n🔀 Router Chain Demo\n")

    # Create simple chains for routing
    from langchain_core.prompts import ChatPromptTemplate

    search_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search assistant. Query: {query}"),
        ("human", "{query}"),
    ])
    search_chain = search_prompt | llm

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization assistant. Text: {query}"),
        ("human", "{query}"),
    ])
    summarize_chain = summarize_prompt | llm

    router = ResearchRouterChain.create_research_router(
        llm=llm,
        search_chain=search_chain,
        summarize_chain=summarize_chain,
    )

    # Test routing
    queries = [
        "Find information about quantum computing",
        "Summarize this article about AI",
    ]

    for query in queries:
        result = router.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Routed to: {result.get('route', 'unknown')}")
        print(f"Result: {result.get('result', 'N/A')[:200]}...\n")


def example_tool_usage() -> None:
    """Demonstrate using tools directly."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Using Tools Directly")
    print("=" * 60)

    # Get OpenRouter-configured LLM for tools that need it
    llm = get_openrouter_llm(temperature=0.7)

    # Search Tool
    print("\n🔍 Search Tool Demo\n")
    search_tool = SearchTool()
    search_results = search_tool.run({"query": "latest artificial intelligence trends"})
    print(f"Search Results:\n{search_results[:500]}...\n")

    # Summarize Tool
    print("\n📝 Summarize Tool Demo\n")
    summarize_tool = SummarizeTool(llm=llm)

    sample_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Large language models like GPT-4 have demonstrated impressive capabilities
    in natural language understanding, code generation, and creative tasks.
    Meanwhile, diffusion models have revolutionized image generation.
    The field continues to evolve rapidly with new architectures and
    applications emerging regularly.
    """

    summary = summarize_tool.run({"text": sample_text, "style": "concise"})
    print(f"Summary:\n{summary}\n")

    # Cite Tool
    print("\n📚 Cite Tool Demo\n")
    cite_tool = CiteTool()

    # Create citations
    citation1 = cite_tool.run({
        "content": "https://example.com/ai-trends-2024",
        "citation_type": "web",
        "metadata": {"title": "AI Trends 2024", "author": "Tech Research Institute"}
    })
    print(f"Citation 1: {citation1}")

    citation2 = cite_tool.run({
        "content": "https://example.com/ml-healthcare",
        "citation_type": "web",
        "metadata": {"title": "ML in Healthcare", "author": "Medical AI Journal"}
    })
    print(f"Citation 2: {citation2}")

    # Get bibliography
    bibliography = cite_tool.get_bibliography("apa")
    print(f"\nBibliography:\n{bibliography}\n")


def example_full_workflow() -> None:
    """Demonstrate a complete research workflow using OpenRouter."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Complete Research Workflow with OpenRouter")
    print("=" * 60)

    # Setup
    setup_logger(log_level="INFO")
    
    # Create agent with OpenRouter
    agent = ResearchAgent.with_openrouter(model="openai/gpt-4-turbo", temperature=0.7)

    # Setup chains for advanced features
    agent.setup_chains()

    # Define research topic
    topic = "Impact of climate change on biodiversity"

    print(f"\n🌍 Starting comprehensive research on: {topic}\n")

    # Start session
    agent.start_session(topic)

    # Multi-query research
    queries = [
        "How does climate change affect species extinction rates?",
        "What are the impacts of rising temperatures on ecosystems?",
        "Which species are most vulnerable to climate change?",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        agent.research(query, generate_report=False)

    # Generate final report
    print("\n📊 Generating comprehensive report...\n")
    report = agent.generate_report()

    # Save report
    output_dir = Path(__file__).parent.parent / "data" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "climate_biodiversity_report.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    print(f"✅ Report saved to: {output_file}")
    print(f"\n📈 Final Session Summary:")
    summary = agent.get_session_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


def example_with_mock_data() -> None:
    """Demonstrate usage without API calls (for testing)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Mock Data Demo (No API Calls)")
    print("=" * 60)

    # This example shows how the tools work with mock data
    # when no API key is configured

    search_tool = SearchTool()

    # This will use mock search if no search engine is configured
    results = search_tool.run({"query": "test query for demo"})
    print(f"\nMock Search Results:\n{results}\n")

    # Transform chain without LLM
    transform_chain = ResearchTransformChain(
        transforms=[
            lambda x: x.upper(),
            lambda x: x.replace("TEST", "DEMO"),
        ]
    )

    result = transform_chain.invoke({"content": "This is a test input"})
    print(f"Transform Result: {result.get('transformed')}\n")


def example_different_models() -> None:
    """Demonstrate using different models via OpenRouter."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Using Different Models via OpenRouter")
    print("=" * 60)

    # Available models on OpenRouter (check openrouter.ai/models for full list)
    models = [
        "openai/gpt-4-turbo",
        "anthropic/claude-3-opus",
        "google/gemini-pro",
    ]

    print("\n📋 Available models to try:\n")
    for model in models:
        print(f"  - {model}")

    print("\n💡 To use a specific model:")
    print("   agent = ResearchAgent.with_openrouter(model='anthropic/claude-3-opus')")
    print("\n   Or with get_openrouter_llm():")
    print("   llm = get_openrouter_llm(model='google/gemini-pro')")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🤖 RESEARCH ASSISTANT - EXAMPLE USAGE")
    print("=" * 60)

    # Check if API keys are available
    settings = get_settings()
    has_openrouter_key = bool(settings.openrouter_api_key)
    has_openai_key = bool(settings.openai_api_key)

    if not has_openrouter_key and not has_openai_key:
        print("\n⚠️  No API key found. Running mock examples only.\n")
        print("💡 Set OPENROUTER_API_KEY in your .env file to use OpenRouter")
        print("   or OPENAI_API_KEY for direct OpenAI access.\n")
        example_with_mock_data()
        example_tool_usage()
        example_different_models()
    else:
        if has_openrouter_key:
            print("\n✅ OpenRouter API key found. Running OpenRouter examples.\n")
            try:
                example_basic_research_openrouter()
                example_chain_usage()
                example_tool_usage()
                example_different_models()
                # Uncomment to run the full workflow example
                # example_full_workflow()
            except Exception as e:
                print(f"\n❌ Example failed: {str(e)}")
                print("\n💡 Make sure you have set up your .env file with API keys.")
        elif has_openai_key:
            print("\n✅ OpenAI API key found. Running direct OpenAI examples.\n")
            try:
                example_basic_research_openai()
                example_chain_usage()
                example_tool_usage()
            except Exception as e:
                print(f"\n❌ Example failed: {str(e)}")
                print("\n💡 Make sure you have set up your .env file with API keys.")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
