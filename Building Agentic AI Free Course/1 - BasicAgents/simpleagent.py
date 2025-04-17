from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

import os
from dotenv import load_dotenv

load_dotenv()

# use groq
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize the language model
agent = Agent(
    model=Gemini(id="gemini-1.5-pro"),
    description="You are an assistant. Please reply based on the question.",
    tools=[DuckDuckGoTools()],
    markdown=True
)

# agent = Agent(
#     model=Groq(id="Gemma2-9b-It"),
#     description="You are an assistant. Please reply based on the question.",
#     tools=[DuckDuckGoTools()],
#     markdown=True
# )

agent.print_response("where is america?")