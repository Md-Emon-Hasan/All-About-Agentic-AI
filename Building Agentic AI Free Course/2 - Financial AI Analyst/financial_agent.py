from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
gemini_llm = Gemini(
    id="gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for accurate financial information",
    model=gemini_llm,
    tools=[DuckDuckGoTools()],
    instructions=[
        "1. Perform targeted searches for financial information",
        "2. Always verify source credibility",
        "3. Include direct links to sources",
        "4. Summarize information concisely"
    ],
    show_tool_calls=True,
    markdown=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=gemini_llm,
    tools=[
        YFinanceTools(
            stock_price=True,
            stock_fundamentals=True,
            analyst_recommendations=True,
            company_news=True
        )
    ],
    instructions=[
        "1. Display data in well-formatted tables",
        "2. Include:",
        "   - Current price and key metrics (P/E, market cap)",
        "   - Analyst consensus (Buy/Hold/Sell)",
        "   - Recent news headlines with dates",
        "3. Highlight significant changes or anomalies"
    ],
    show_tool_calls=True,
    markdown=True
)

# Multi-Agent Team
research_assistant = Agent(
    name="Financial Research Assistant",
    model=gemini_llm,
    team=[web_search_agent, finance_agent],
    instructions=[
        "1. Combine web and financial data into a cohesive report",
        "2. Structure output with clear sections:",
        "   - [Latest News] (from web search)",
        "   - [Stock Analysis] (from financial data)",
        "3. Use professional tone suitable for investors",
        "4. Include all relevant sources and timestamps"
    ],
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    research_assistant.print_response(
        "Provide a comprehensive analysis of NVIDIA (NVDA) including: "
        "1. Current stock performance and key metrics\n"
        "2. Analyst consensus and price targets\n"
        "3. Latest company news and industry developments",
        stream=True
    )
