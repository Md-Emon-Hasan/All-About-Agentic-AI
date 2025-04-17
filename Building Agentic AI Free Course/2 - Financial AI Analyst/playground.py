# import openai
from agno.agent import Agent
import agno.api
# from phi.model.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from agno.models.groq import Groq

import os
import agno
from agno.playground import Playground, serve_playground_app
# Load environment variables from .env file
load_dotenv()

agno.api=os.getenv("PHI_API_KEY")

## web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="qwen-qwq-32b"),
    tools=[DuckDuckGoTools()],
    instructions=["Alway include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="qwen-qwq-32b"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)