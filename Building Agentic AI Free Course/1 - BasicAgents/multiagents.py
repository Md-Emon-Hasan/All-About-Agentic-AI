from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv

load_dotenv()

# set api keys
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# web agent using groq model
web_agent = Agent(
    name='Web Agent',
    role='search the web for information',
    model=Groq(id='qwen-qwq-32b'),
    tools=[DuckDuckGoTools()],
    instructions='You are a web agent. You can search the web for information.',
    show_tool_calls=True,
    markdown=True
)

# financial agent using google gemini model
financial_agent = Agent(
    name='Financial Agent',
    role='answer questions about the stock market',
    model=Gemini(id='gemini-1.5-pro'),
    tools=[YFinanceTools(
        stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True
            )],
    instructions='You are a financial agent. You can answer questions about the stock market.',
    show_tool_calls=True,
    markdown=True
)

agent_team = Agent(
    team=[web_agent, financial_agent],
    model=Gemini(id='gemini-1.5-pro'),
    instructions=['You are a team of agents. You can answer questions about the stock market and search the web for information.'],
    show_tool_calls=True,
    markdown=True
)

agent_team.print_response("Give a simple comparison of Tesla, NVDA, and Apple stocks for long-term investment.")