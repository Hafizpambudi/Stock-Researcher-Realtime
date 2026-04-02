#!/usr/bin/env python3
"""
Browser Tool Examples.

This module demonstrates how to use the BrowserTool both standalone
and integrated with the ResearchAgent.

Usage:
    python examples/browser_example.py

Note: This example uses mock data for demonstration. For real web browsing,
ensure you have an internet connection and respect website terms of service.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.research_agent import ResearchAgent
from src.tools.browser_tool import BrowserTool, BrowserResult


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def example_standalone_browser() -> None:
    """Demonstrate standalone BrowserTool usage."""
    print_section("Example 1: Standalone BrowserTool")

    # Initialize the browser tool
    browser = BrowserTool(cache_size=50, requests_per_second=2.0)

    print("\n1. Browser Tool Initialization")
    print(f"   - Cache size: {browser._cache.max_size}")
    print(f"   - Rate limit: {browser._rate_limiter.requests_per_second} req/s")

    # Note: For real usage, uncomment the following examples:
    print("\n2. Visit a URL (Example - commented for safety)")
    print("""
    # Visit a URL and get content
    result = browser.visit("https://example.com")
    if result.success:
        print(f"Title: {result.title}")
        print(f"Content preview: {result.content[:200]}...")
        print(f"Links found: {len(result.links)}")
    """)

    print("\n3. Extract Clean Content (Example - commented for safety)")
    print("""
    # Extract main content (removes ads, navigation, etc.)
    result = browser.extract_content("https://example.com/article")
    if result.success:
        print(f"Clean content: {result.content[:500]}...")
    """)

    print("\n4. Get Links from Page (Example - commented for safety)")
    print("""
    # Get all links from a page
    result = browser.get_links("https://example.com")
    if result.success:
        for link in result.links[:10]:
            print(f"  - {link}")
    """)

    print("\n5. Web Search (Example - commented for safety)")
    print("""
    # Search the web
    results = browser.search("Python programming")
    print(results)
    """)

    print("\n6. Cache Statistics")
    stats = browser.get_cache_stats()
    print(f"   - Cached URLs: {stats['cached_urls']}")
    print(f"   - Max cache size: {stats['max_size']}")
    print(f"   - Total cache hits: {stats['total_hits']}")

    print("\n7. Clear Cache")
    browser.clear_cache()
    print("   Cache cleared successfully")


def example_browser_with_agent() -> None:
    """Demonstrate BrowserTool integrated with ResearchAgent."""
    print_section("Example 2: BrowserTool with ResearchAgent")

    # Initialize the agent (browser tool is included by default)
    print("\n1. Initialize ResearchAgent with BrowserTool")
    print("""
    from src.agents.research_agent import ResearchAgent
    
    # Create agent - browser tool is included automatically
    agent = ResearchAgent.with_openrouter()
    
    # Or with custom browser tool settings
    from src.tools.browser_tool import BrowserTool
    custom_browser = BrowserTool(cache_size=100, requests_per_second=1.0)
    agent = ResearchAgent.with_openrouter(browser_tool=custom_browser)
    """)

    print("\n2. Browse URL with Agent (Example - commented for safety)")
    print("""
    # Start a research session
    agent.start_session("Web Development Trends")
    
    # Browse a specific URL
    result = agent.browse_url("https://example.com/article")
    
    if result["success"]:
        print(f"Title: {result['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"Links found: {len(result['links'])}")
        print(f"From cache: {result.get('from_cache', False)}")
    else:
        print(f"Error: {result.get('error')}")
    """)

    print("\n3. Agent with Browser in Research Workflow (Example)")
    print("""
    # The agent can now use the browser tool during research
    # When you ask it to research a topic, it may:
    # 1. Search for relevant URLs
    # 2. Browse specific pages to extract content
    # 3. Summarize the findings
    # 4. Create citations
    
    report = agent.research(
        "Latest trends in web development",
        generate_report=True
    )
    
    print(report.to_markdown())
    """)


def example_browser_features() -> None:
    """Demonstrate specific BrowserTool features."""
    print_section("Example 3: BrowserTool Features")

    browser = BrowserTool()

    print("\n1. LRU Caching")
    print("""
    # The browser tool caches fetched URLs to avoid redundant requests
    browser = BrowserTool(cache_size=100)
    
    # First visit - fetches from web
    result1 = browser.visit("https://example.com")
    print(result1.from_cache)  # False
    
    # Second visit - uses cache
    result2 = browser.visit("https://example.com")
    print(result2.from_cache)  # True
    
    # Check cache stats
    stats = browser.get_cache_stats()
    print(f"Cached URLs: {stats['cached_urls']}")
    """)

    print("\n2. Rate Limiting")
    print("""
    # Prevent overwhelming servers with rate limiting
    browser = BrowserTool(requests_per_second=2.0)  # Max 2 requests/second
    
    # Requests are automatically delayed to respect the limit
    browser.visit("https://example.com/page1")
    browser.visit("https://example.com/page2")  # Waits if needed
    """)

    print("\n3. Error Handling")
    print("""
    # The tool handles various errors gracefully
    result = browser.visit("https://nonexistent-domain-12345.com")
    
    if not result.success:
        print(f"Error: {result.error}")
        # Possible errors:
        # - "Invalid URL"
        # - "Request timed out"
        # - "Connection failed"
        # - HTTP error messages
    """)

    print("\n4. Content Extraction with Readability")
    print("""
    # Extract main content, removing ads and navigation
    result = browser.extract_content("https://example.com/article")
    
    # Uses readability-lxml for clean content extraction
    print(result.content)  # Main article content only
    """)

    print("\n5. Metadata Extraction")
    print("""
    # Get detailed metadata about a page
    result = browser.visit("https://example.com")
    
    print(result.metadata)
    # Includes:
    # - URL, status code, content type
    # - Meta tags (description, keywords, etc.)
    # - Open Graph data (og:title, og:image, etc.)
    # - Fetch timestamp
    """)

    print("\n6. BrowserResult Structure")
    print("""
    # The visit() method returns a BrowserResult with:
    result = browser.visit("https://example.com")
    
    print(f"URL: {result.url}")
    print(f"Title: {result.title}")
    print(f"Content: {result.content[:200]}...")
    print(f"Links: {len(result.links)} found")
    print(f"Metadata: {result.metadata}")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Status Code: {result.status_code}")
    print(f"Fetch Time: {result.fetch_time:.2f}s")
    print(f"From Cache: {result.from_cache}")
    """)


def example_tool_actions() -> None:
    """Demonstrate using the tool with different actions."""
    print_section("Example 4: Tool Actions")

    print("""
The BrowserTool supports multiple actions via the run() method:

from src.tools.browser_tool import BrowserTool

browser = BrowserTool()

# 1. Visit a URL (default action)
result = browser.run({"url": "https://example.com", "action": "visit"})

# 2. Search the web
result = browser.run({"url": "Python tutorials", "action": "search"})

# 3. Extract clean content
result = browser.run({"url": "https://example.com/article", "action": "extract"})

# 4. Get all links
result = browser.run({"url": "https://example.com", "action": "links"})

# The action parameter determines which method is called internally
""")


def example_best_practices() -> None:
    """Demonstrate best practices for using the BrowserTool."""
    print_section("Example 5: Best Practices")

    print("""
1. RESPECT WEBSITES
   - Always respect robots.txt files
   - Don't scrape sites that prohibit it
   - Add delays between requests (use rate limiting)
   
   browser = BrowserTool(requests_per_second=1.0)  # Conservative rate

2. USE CACHING
   - Enable caching to avoid redundant requests
   - Clear cache when needed
   
   browser = BrowserTool(cache_size=100)
   # ... use browser ...
   browser.clear_cache()  # Clear when needed

3. HANDLE ERRORS
   - Always check result.success before using data
   
   result = browser.visit(url)
   if result.success:
       process_content(result.content)
   else:
       logger.error(f"Failed: {result.error}")

4. EXTRACT CLEAN CONTENT
   - Use extract_content() for article pages
   - Use visit() for general pages
   
   # For articles
   result = browser.extract_content("https://example.com/article")
   
   # For general pages
   result = browser.visit("https://example.com")

5. MONITOR CACHE
   - Check cache stats to understand usage patterns
   
   stats = browser.get_cache_stats()
   print(f"Cache hits: {stats['total_hits']}")

6. USE PROPER USER AGENT
   - The tool uses a standard browser User-Agent by default
   - This helps avoid being blocked

7. RESPECT RATE LIMITS
   - Don't set requests_per_second too high
   - Consider the server's capacity
   
   browser = BrowserTool(requests_per_second=2.0)  # Reasonable limit
""")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" BROWSER TOOL EXAMPLES")
    print(" Intelligent Research Assistant")
    print("=" * 60)

    example_standalone_browser()
    example_browser_with_agent()
    example_browser_features()
    example_tool_actions()
    example_best_practices()

    print_section("Examples Complete")
    print("""
These examples demonstrate the BrowserTool capabilities.

To use with real URLs:
1. Uncomment the example code blocks
2. Replace example URLs with real ones
3. Ensure you have internet connectivity
4. Respect website terms of service

For more information, see the README.md documentation.
""")


if __name__ == "__main__":
    main()
