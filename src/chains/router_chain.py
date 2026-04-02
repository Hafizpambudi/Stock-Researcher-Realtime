"""
Router Chain implementation for the Research Assistant.

This module implements a router chain that dynamically routes input
to different downstream chains based on the input content or intent.
"""

from typing import Any, Optional

from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RouteResult:
    """Represents the result of a routing decision."""

    def __init__(
        self,
        route_name: str,
        chain: Runnable,
        confidence: float = 1.0,
        reasoning: str = "",
    ) -> None:
        """
        Initialize a RouteResult.

        Args:
            route_name: The name of the selected route.
            chain: The chain to execute for this route.
            confidence: Confidence score for the routing decision.
            reasoning: Explanation for why this route was selected.
        """
        self.route_name = route_name
        self.chain = chain
        self.confidence = confidence
        self.reasoning = reasoning


class ResearchRouterChain(Chain):
    """
    A router chain for dynamically routing research tasks to appropriate handlers.

    This chain analyzes the input and routes it to the most appropriate
    downstream chain based on the content type, intent, or other criteria.
    It's useful for handling diverse research queries that require different
    processing strategies.

    Attributes:
        name: The name of the chain.
        routes: A dictionary mapping route names to their chains.
        router_llm: The language model used for routing decisions.
        input_key: The key for the chain's input.
        output_key: The key for the chain's output.

    Example:
        >>> router = ResearchRouterChain(
        ...     routes={
        ...         "search": search_chain,
        ...         "analyze": analysis_chain,
        ...         "summarize": summary_chain,
        ...     },
        ...     router_llm=my_llm
        ... )
        >>> result = router.invoke({"query": "Find recent AI news"})
    """

    name: str = "research_router_chain"
    input_key: str = "query"
    output_key: str = "result"

    routes: dict[str, Runnable] = Field(default_factory=dict)
    router_llm: Optional[BaseLanguageModel] = Field(default=None)
    default_route: Optional[str] = Field(default=None)
    route_descriptions: dict[str, str] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        routes: Optional[dict[str, Runnable]] = None,
        router_llm: Optional[BaseLanguageModel] = None,
        default_route: Optional[str] = None,
        route_descriptions: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Router Chain.

        Args:
            routes: Dictionary mapping route names to chains.
            router_llm: Language model for routing decisions.
            default_route: Default route name if no match is found.
            route_descriptions: Descriptions of each route for the router LLM.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            routes=routes or {},
            router_llm=router_llm,
            default_route=default_route,
            route_descriptions=route_descriptions or {},
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        """Return the expected input keys for the chain."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys produced by the chain."""
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[Callbacks] = None,
    ) -> dict[str, Any]:
        """
        Execute the router chain with the given inputs.

        Args:
            inputs: Dictionary containing the input values.
            run_manager: Optional callback manager for execution tracking.

        Returns:
            Dictionary containing the output from the selected route.

        Raises:
            ValueError: If no routes are configured or routing fails.
        """
        if not self.routes:
            raise ValueError("No routes configured for router chain")

        query = inputs.get(self.input_key, "")
        logger.info(f"Routing query: {query[:100]}...")

        try:
            # Determine the appropriate route
            route_result = self._route(query)

            logger.info(f"Routed to: {route_result.route_name}")

            # Execute the selected chain
            selected_chain = route_result.chain
            result = selected_chain.invoke(
                {self.input_key: query}, config={"callbacks": run_manager}
            )

            # Extract the result
            if isinstance(result, dict):
                final_result = result.get(self.output_key, result.get("result", str(result)))
            else:
                final_result = str(result)

            return {
                self.output_key: final_result,
                "route": route_result.route_name,
                "confidence": route_result.confidence,
            }

        except Exception as e:
            logger.error(f"Router chain execution failed: {str(e)}")
            raise

    def _route(self, query: str) -> RouteResult:
        """
        Determine the appropriate route for the given query.

        Args:
            query: The input query to route.

        Returns:
            A RouteResult containing the selected route and chain.
        """
        if not self.router_llm:
            # Use keyword-based routing if no LLM is available
            return self._keyword_route(query)

        # Use LLM-based routing
        return self._llm_route(query)

    def _keyword_route(self, query: str) -> RouteResult:
        """
        Route based on keyword matching.

        Args:
            query: The input query.

        Returns:
            A RouteResult with the matched route.
        """
        query_lower = query.lower()

        # Define keyword mappings
        keyword_routes = {
            "search": ["search", "find", "look up", "discover", "get information"],
            "summarize": ["summarize", "summary", "brief", "overview", "condense"],
            "analyze": ["analyze", "analysis", "examine", "evaluate", "assess"],
            "compare": ["compare", "versus", "vs", "difference", "similar"],
            "explain": ["explain", "what is", "describe", "tell me about"],
        }

        # Find matching route
        for route_name, keywords in keyword_routes.items():
            if route_name in self.routes:
                for keyword in keywords:
                    if keyword in query_lower:
                        return RouteResult(
                            route_name=route_name,
                            chain=self.routes[route_name],
                            confidence=0.8,
                            reasoning=f"Matched keyword: {keyword}",
                        )

        # Return default route if available
        if self.default_route and self.default_route in self.routes:
            return RouteResult(
                route_name=self.default_route,
                chain=self.routes[self.default_route],
                confidence=0.5,
                reasoning="No keyword match, using default route",
            )

        # Return first available route as fallback
        first_route = next(iter(self.routes.keys()))
        return RouteResult(
            route_name=first_route,
            chain=self.routes[first_route],
            confidence=0.3,
            reasoning="Fallback to first route",
        )

    def _llm_route(self, query: str) -> RouteResult:
        """
        Route based on LLM analysis of the query.

        Args:
            query: The input query.

        Returns:
            A RouteResult with the LLM-selected route.
        """
        # Build route descriptions for the prompt
        route_info = "\n".join(
            [
                f"- {name}: {self.route_descriptions.get(name, 'No description')}"
                for name in self.routes.keys()
            ]
        )

        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a routing assistant for a research system. "
                    "Analyze the user's query and select the most appropriate route.\n\n"
                    "Available routes:\n"
                    f"{route_info}\n\n"
                    "Respond with ONLY the route name, nothing else.",
                ),
                ("human", "Query: {query}"),
            ]
        )

        try:
            prompt = router_prompt.format(query=query)
            response = self.router_llm.invoke(prompt)
            selected_route = (
                response.content if hasattr(response, "content") else str(response)
            ).strip().lower()

            # Find matching route (case-insensitive)
            for route_name in self.routes.keys():
                if route_name.lower() in selected_route:
                    return RouteResult(
                        route_name=route_name,
                        chain=self.routes[route_name],
                        confidence=0.9,
                        reasoning=f"LLM selected: {selected_route}",
                    )

            # Fallback to default
            if self.default_route:
                return RouteResult(
                    route_name=self.default_route,
                    chain=self.routes[self.default_route],
                    confidence=0.5,
                    reasoning=f"LLM output '{selected_route}' not matched, using default",
                )

        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}, falling back to keyword routing")

        # Fallback to keyword routing
        return self._keyword_route(query)

    def add_route(self, name: str, chain: Runnable, description: str = "") -> None:
        """
        Add a route to the router.

        Args:
            name: The name of the route.
            chain: The chain to execute for this route.
            description: Description of when to use this route.
        """
        self.routes[name] = chain
        if description:
            self.route_descriptions[name] = description
        logger.info(f"Added route: {name}")

    def remove_route(self, name: str) -> None:
        """
        Remove a route from the router.

        Args:
            name: The name of the route to remove.
        """
        if name in self.routes:
            del self.routes[name]
            self.route_descriptions.pop(name, None)
            logger.info(f"Removed route: {name}")

    def get_available_routes(self) -> list[str]:
        """
        Get the list of available route names.

        Returns:
            A list of route names.
        """
        return list(self.routes.keys())

    @classmethod
    def create_research_router(
        cls,
        llm: BaseLanguageModel,
        search_chain: Optional[Runnable] = None,
        summarize_chain: Optional[Runnable] = None,
        analyze_chain: Optional[Runnable] = None,
        compare_chain: Optional[Runnable] = None,
    ) -> "ResearchRouterChain":
        """
        Create a pre-configured research router with common routes.

        Args:
            llm: The language model for routing decisions.
            search_chain: Chain for search operations.
            summarize_chain: Chain for summarization operations.
            analyze_chain: Chain for analysis operations.
            compare_chain: Chain for comparison operations.

        Returns:
            A configured ResearchRouterChain instance.
        """
        routes = {}
        descriptions = {}

        if search_chain:
            routes["search"] = search_chain
            descriptions["search"] = "Use for finding information, looking up facts, or discovering new content."

        if summarize_chain:
            routes["summarize"] = summarize_chain
            descriptions["summarize"] = "Use for creating summaries, overviews, or condensed versions of content."

        if analyze_chain:
            routes["analyze"] = analyze_chain
            descriptions["analyze"] = "Use for examining, evaluating, or assessing information in detail."

        if compare_chain:
            routes["compare"] = compare_chain
            descriptions["compare"] = "Use for comparing items, finding differences, or analyzing similarities."

        return cls(
            routes=routes,
            router_llm=llm,
            route_descriptions=descriptions,
            default_route="search" if "search" in routes else None,
        )
