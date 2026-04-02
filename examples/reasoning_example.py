"""
OpenRouter Reasoning Feature Example

This example demonstrates how to use the OpenRouter reasoning feature
with the Research Agent. The reasoning feature provides step-by-step
reasoning traces for supported models.

Supported Models:
- minimax/minimax-m2.5
- Some Anthropic Claude models (check OpenRouter for current support)
- Other reasoning-capable models as they become available

Usage:
    python examples/reasoning_example.py

Note:
    - Requires OPENROUTER_API_KEY environment variable
    - Set OPENROUTER_REASONING_ENABLED=true to enable via environment
    - Or use reasoning_enabled=True in code
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.research_agent import ResearchAgent
from src.utils.config import get_openrouter_llm, get_settings


class ReasoningCallback(BaseCallbackHandler):
    """
    Callback handler to capture and display reasoning traces.

    This callback captures the raw response from the LLM and attempts
    to extract reasoning information from the response metadata.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the callback handler."""
        self.verbose = verbose
        self.reasoning_traces = []
        self.response_metadata = []

    def on_llm_end(self, response, **kwargs):
        """
        Called when the LLM finishes generating a response.

        Args:
            response: The LLM response object.
            **kwargs: Additional keyword arguments.
        """
        # Store response metadata for inspection
        self.response_metadata.append(response.response_metadata)

        if self.verbose:
            print("\n--- Response Metadata ---")
            print(f"Keys: {list(response.response_metadata.keys())}")

            # Try to extract reasoning from various possible locations
            if "reasoning" in response.response_metadata:
                reasoning = response.response_metadata["reasoning"]
                self.reasoning_traces.append(reasoning)
                print(f"\nReasoning: {reasoning}")

            # OpenRouter may include reasoning in different fields
            for key, value in response.response_metadata.items():
                if "reason" in key.lower():
                    print(f"\n{key}: {value}")
                    self.reasoning_traces.append(value)

    def on_chat_model_end(self, response, **kwargs):
        """
        Called when a chat model finishes generating a response.

        Args:
            response: The chat model response object.
            **kwargs: Additional keyword arguments.
        """
        if self.verbose:
            print("\n--- Chat Model Response ---")
            print(f"Response metadata: {response.response_metadata}")


def example_1_basic_reasoning():
    """
    Example 1: Basic usage with reasoning enabled.

    Demonstrates how to create an agent with reasoning enabled
    and make a simple query.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Reasoning Usage")
    print("=" * 60)

    # Create agent with reasoning enabled
    # Using a model that supports reasoning
    agent = ResearchAgent.with_openrouter(
        model="minimax/minimax-m2.5",  # or another reasoning-supported model
        reasoning_enabled=True,
        temperature=0.7,
    )

    print(f"\nAgent created with reasoning_enabled={agent.reasoning_enabled}")
    print(f"Model: {agent.llm.model_name}")

    # Start a research session
    agent.start_session("Understanding reasoning feature")

    # Make a query that benefits from reasoning
    query = "Explain step by step: What is 15% of 250?"

    print(f"\nQuery: {query}")
    print("\n--- Response ---")

    try:
        # Execute the query
        result = agent.agent_executor.invoke({"input": query})
        print(f"\nAnswer: {result.get('output', 'No response')}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This may fail if the model doesn't support reasoning or API key is not set.")


def example_2_reasoning_with_callback():
    """
    Example 2: Using callbacks to capture reasoning traces.

    Demonstrates how to use LangChain callbacks to capture and
    inspect reasoning information from the response.
    """
    print("\n" + "=" * 60)
    print("Example 2: Reasoning with Callbacks")
    print("=" * 60)

    # Create callback handler
    reasoning_callback = ReasoningCallback(verbose=True)

    # Create LLM with reasoning enabled and callback
    llm = get_openrouter_llm(
        model="minimax/minimax-m2.5",
        reasoning_enabled=True,
        callbacks=[reasoning_callback],
    )

    # Create agent with the custom LLM
    agent = ResearchAgent(llm=llm)

    print(f"\nAgent created with reasoning callback")

    # Start session
    agent.start_session("Reasoning with callbacks")

    # Make a query
    query = "If a train travels at 60 mph for 2.5 hours, how far does it go? Show your work."

    print(f"\nQuery: {query}")
    print("\n--- Processing ---")

    try:
        result = agent.agent_executor.invoke({"input": query})
        print(f"\nAnswer: {result.get('output', 'No response')}")

        # Check captured reasoning
        if reasoning_callback.reasoning_traces:
            print("\n--- Captured Reasoning Traces ---")
            for i, trace in enumerate(reasoning_callback.reasoning_traces, 1):
                print(f"Trace {i}: {trace}")
    except Exception as e:
        print(f"Error: {e}")


def example_3_multi_turn_conversation():
    """
    Example 3: Multi-turn conversation with reasoning preservation.

    Demonstrates how reasoning works across multiple turns of conversation,
    maintaining context and reasoning state.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)

    # Create agent with reasoning enabled
    agent = ResearchAgent.with_openrouter(
        model="minimax/minimax-m2.5",
        reasoning_enabled=True,
    )

    # Start session
    agent.start_session("Multi-turn reasoning conversation")

    # Conversation turns
    conversation = [
        "What is the capital of France?",
        "What is the population of that city?",
        "How does that compare to London's population?",
    ]

    print("\nStarting multi-turn conversation...\n")

    for i, query in enumerate(conversation, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {query}")

        try:
            result = agent.agent_executor.invoke({
                "input": query,
                "chat_history": agent.session.queries if agent.session else [],
            })
            print(f"Assistant: {result.get('output', 'No response')[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

        print()


def example_4_comparison_with_without_reasoning():
    """
    Example 4: Compare responses with and without reasoning.

    Demonstrates the difference between using reasoning enabled vs disabled.
    """
    print("\n" + "=" * 60)
    print("Example 4: Comparison (With vs Without Reasoning)")
    print("=" * 60)

    query = "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

    print(f"\nQuery: {query}")
    print("\nThis is a classic cognitive reflection test!")

    # Without reasoning
    print("\n--- Without Reasoning ---")
    agent_no_reasoning = ResearchAgent.with_openrouter(
        model="minimax/minimax-m2.5",
        reasoning_enabled=False,
    )
    agent_no_reasoning.start_session("Without reasoning")

    try:
        result = agent_no_reasoning.agent_executor.invoke({"input": query})
        print(f"Response: {result.get('output', 'No response')[:300]}...")
    except Exception as e:
        print(f"Error: {e}")

    # With reasoning
    print("\n--- With Reasoning ---")
    agent_with_reasoning = ResearchAgent.with_openrouter(
        model="minimax/minimax-m2.5",
        reasoning_enabled=True,
    )
    agent_with_reasoning.start_session("With reasoning")

    try:
        result = agent_with_reasoning.agent_executor.invoke({"input": query})
        print(f"Response: {result.get('output', 'No response')[:300]}...")
    except Exception as e:
        print(f"Error: {e}")


def example_5_environment_variable():
    """
    Example 5: Using environment variable to enable reasoning.

    Demonstrates how to enable reasoning via the OPENROUTER_REASONING_ENABLED
    environment variable instead of code.
    """
    print("\n" + "=" * 60)
    print("Example 5: Environment Variable Configuration")
    print("=" * 60)

    # Show current settings
    settings = get_settings()
    print(f"\nCurrent OPENROUTER_REASONING_ENABLED setting: {settings.openrouter_reasoning_enabled}")

    # Demonstrate how to set via environment
    print("\nTo enable via environment variable:")
    print("  export OPENROUTER_REASONING_ENABLED=true  # Unix/macOS")
    print("  set OPENROUTER_REASONING_ENABLED=true     # Windows")
    print("\nThen create agent normally:")
    print("  agent = ResearchAgent.with_openrouter()")

    # Create agent (will use env var setting)
    agent = ResearchAgent.with_openrouter()
    print(f"\nAgent reasoning_enabled: {agent.reasoning_enabled}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OpenRouter Reasoning Feature Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  Warning: OPENROUTER_API_KEY not set!")
        print("Set your API key before running these examples:")
        print("  export OPENROUTER_API_KEY=your_key_here  # Unix/macOS")
        print("  set OPENROUTER_API_KEY=your_key_here     # Windows")
        print("\nContinuing with demonstration mode...\n")

    # Run examples
    examples = [
        ("Basic Reasoning Usage", example_1_basic_reasoning),
        ("Reasoning with Callbacks", example_2_reasoning_with_callback),
        ("Multi-turn Conversation", example_3_multi_turn_conversation),
        ("Comparison (With vs Without)", example_4_comparison_with_without_reasoning),
        ("Environment Variable", example_5_environment_variable),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nFor more information, see the README.md section on OpenRouter Reasoning.")
    print("Supported models and features may change - check OpenRouter documentation.")


if __name__ == "__main__":
    main()
