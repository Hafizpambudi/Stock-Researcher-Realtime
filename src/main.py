"""
Main entry point for the Research Assistant.

This module provides the CLI interface and main function for running
the Research Assistant application.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

from src.agents.research_agent import ResearchAgent
from src.utils.config import get_settings, validate_environment
from src.utils.logger import setup_logger


def create_llm(model: Optional[str] = None, temperature: float = 0.7) -> ChatOpenAI:
    """
    Create a language model instance.

    Args:
        model: The model name to use. If None, uses config default.
        temperature: The temperature for generation.

    Returns:
        A ChatOpenAI instance.
    """
    settings = get_settings()
    model_name = model or settings.openai_model

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
    )


def run_research(
    topic: str,
    model: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Run a research session on the given topic.

    Args:
        topic: The research topic.
        model: Optional model name override.
        output_file: Optional file to save the report.
        verbose: Whether to enable verbose output.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)

    # Validate environment
    is_valid, missing = validate_environment()
    if not is_valid:
        print(f"Error: Missing required environment variables: {missing}")
        print("Please copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    # Create LLM and agent
    llm = create_llm(model)
    agent = ResearchAgent(llm=llm)

    # Setup chains for advanced routing
    agent.setup_chains()

    print(f"\n🔍 Starting research on: {topic}\n")

    # Run research
    try:
        report = agent.research(topic)

        # Display report
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60 + "\n")
        print(report.to_markdown())

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            print(f"\n📄 Report saved to: {output_file}")

        # Print session summary
        summary = agent.get_session_summary()
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Session ID: {summary.get('session_id', 'N/A')}")
        print(f"Queries: {summary.get('queries_count', 0)}")
        print(f"Findings: {summary.get('findings_count', 0)}")
        print(f"Citations: {summary.get('citations_count', 0)}")

    except Exception as e:
        print(f"\n❌ Research failed: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_interactive(model: Optional[str] = None) -> None:
    """
    Run an interactive research session.

    Args:
        model: Optional model name override.
    """
    # Validate environment
    is_valid, missing = validate_environment()
    if not is_valid:
        print(f"Error: Missing required environment variables: {missing}")
        sys.exit(1)

    # Setup
    setup_logger(log_level="INFO")
    llm = create_llm(model)
    agent = ResearchAgent(llm=llm)
    agent.setup_chains()

    print("\n" + "=" * 60)
    print("🤖 Intelligent Research Assistant")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'report' to generate a report of current findings.")
    print("Type 'reset' to start a new research session.\n")

    while True:
        try:
            user_input = input("📝 Research query: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Ending research session. Goodbye!")
                break

            if user_input.lower() == "report":
                if agent.session:
                    report = agent.generate_report()
                    print("\n" + report.to_markdown())
                else:
                    print("\n⚠️  No active session. Start with a research query first.")
                continue

            if user_input.lower() == "reset":
                agent.reset()
                print("\n🔄 Session reset. Ready for new research topic.")
                continue

            # Run research
            print("\n🔍 Researching...\n")
            report = agent.research(user_input, generate_report=False)
            print(report.detailed_analysis[:1000] + "..." if len(report.detailed_analysis) > 1000 else report.detailed_analysis)
            print("\n💡 Type 'report' to generate a full report, or continue researching.\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Intelligent Research Assistant - Powered by LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research a topic and save report
  %(prog)s research "Latest AI developments" -o report.md

  # Interactive research session
  %(prog)s interactive

  # Research with specific model
  %(prog)s research "Climate change solutions" --model gpt-4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Research command
    research_parser = subparsers.add_parser(
        "research", help="Research a specific topic"
    )
    research_parser.add_argument(
        "topic", type=str, help="The research topic or query"
    )
    research_parser.add_argument(
        "-o", "--output", type=str, help="Output file for the report"
    )
    research_parser.add_argument(
        "-m", "--model", type=str, help="Model to use (default: from config)"
    )
    research_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start an interactive research session"
    )
    interactive_parser.add_argument(
        "-m", "--model", type=str, help="Model to use (default: from config)"
    )

    args = parser.parse_args()

    if args.command == "research":
        run_research(
            topic=args.topic,
            model=args.model,
            output_file=args.output,
            verbose=args.verbose,
        )
    elif args.command == "interactive":
        run_interactive(model=args.model)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
