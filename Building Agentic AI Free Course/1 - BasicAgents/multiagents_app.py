import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import io
import sys
import re

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Sub-agents
web_agent = Agent(
    name='Web Agent',
    role='search the web for information',
    model=Groq(id='qwen-qwen-32b'),
    tools=[DuckDuckGoTools()],
    instructions='You are a web agent. You can search the web for any kind of query.',
    show_tool_calls=True,
    markdown=True
)

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
    instructions='You are a financial agent. Answer based on stock market data.',
    show_tool_calls=True,
    markdown=True
)

# Main team agent
agent_team = Agent(
    team=[web_agent, financial_agent],
    model=Gemini(id='gemini-1.5-pro'),
    instructions=['You are a team of agents. Use your teammates to answer queries based on their expertise.'],
    show_tool_calls=True,
    markdown=True
)

# Streamlit UI
st.set_page_config(page_title="Multi-Agent Gemini+Groq App", layout="centered")
st.title("Multi-Agent AI Assistant")
st.markdown("Ask anything about the **stock market** or **search the web**")

user_input = st.text_input("Enter your query:", placeholder="E.g., Compare Tesla, Apple and Nvidia stocks")

if st.button("Run Query") and user_input.strip():
    st.info("Query: " + user_input)

    # Capture stdout from agent_team.print_response
    with st.spinner("Agents are thinking..."):
        buffer = io.StringIO()
        sys.stdout = buffer  # Redirect print output to buffer

        agent_team.print_response(user_input)

        sys.stdout = sys.__stdout__  # Restore stdout
        result = buffer.getvalue()

        # Remove ANSI escape codes using regex
        cleaned_result = re.sub(r'\x1b\[[0-9;]*m', '', result)

        st.markdown("###Response")
        st.markdown(cleaned_result)
