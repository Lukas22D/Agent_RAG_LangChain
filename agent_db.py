from langchain.agents import load_tools, initialize_agent, AgentType, Tool
import requests
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.utilities import ArxivAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatCohere(model='command-r-plus', cohere_api_key=os.getenv('COHERE_API_KEY'))
arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name='arxiv_tool',
    description='Search on arxiv. The tool can search a keyword on arxiv for the top papers. It will return publishing date, title, and abstract.',
    func=arxiv.run
)

tools = [arxiv_tool]

agent_chain = initialize_agent(tools, llm, Agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
response =agent_chain.run("What is a LLM?")
print(f'RESPONSE:', response)
