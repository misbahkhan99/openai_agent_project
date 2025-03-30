import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv
import os
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up the page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Misbah's AI Assistant")

# Initialize the agent (only once)
@st.cache_resource
def initialize_agent():
    load_dotenv()
    set_tracing_disabled(disabled=True)
    
    provider = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash-exp",
        openai_client=provider,
    )

    return Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,
    )

agent = initialize_agent()

# Create a new event loop for Streamlit
def run_query(agent, query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = Runner.run_sync(agent, query)
        return result
    finally:
        loop.close()

# User input
user_input = st.text_input("Ask me anything:", placeholder=" ")

if user_input:
    with st.spinner("Thinking..."):
        result = run_query(agent, user_input)
        st.write(result.final_output)