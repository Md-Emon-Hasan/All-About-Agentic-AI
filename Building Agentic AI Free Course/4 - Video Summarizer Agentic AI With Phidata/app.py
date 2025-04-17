import streamlit as st 
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 1.5 Flash")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-1.5-pro"),
        tools=[DuckDuckGoTools()],
        markdown=True,
    )

# Initialize the agent
multimodal_agent = initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file", 
    type=['mp4', 'mov', 'avi'], 
    help="Upload a video for AI analysis"
)

if video_file:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    
    st.video(video_path)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content...",
        help="Provide specific questions about the video."
    )

    if st.button("🔍 Analyze Video"):
        if not user_query:
            st.warning("Please enter a question about the video.")
        else:
            try:
                with st.spinner("Analyzing video and gathering insights..."):
                    # Create the prompt with video context
                    prompt = f"""
                    Analyze the following video and answer the user's question.
                    Video content: {video_path}
                    User question: {user_query}
                    
                    Provide a detailed response with:
                    1. Key insights from the video
                    2. Relevant context from web research (if needed)
                    3. Clear timestamps for important moments
                    """
                    
                    # Get response from agent
                    response = multimodal_agent.run(prompt)
                
                # Display results
                st.subheader("Analysis Results")
                st.markdown(response.content)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                # Clean up temporary file
                try:
                    Path(video_path).unlink(missing_ok=True)
                except:
                    pass

# Custom CSS for text area
st.markdown("""
<style>
.stTextArea textarea {
    height: 100px;
}
</style>
""", unsafe_allow_html=True)