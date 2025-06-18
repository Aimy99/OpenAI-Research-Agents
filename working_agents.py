import os
from dotenv import load_dotenv
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests


# Openai Imports
from openai import AsyncOpenAI
from agents import Agent,  OpenAIChatCompletionsModel, function_tool, set_tracing_disabled


# loading env variables.
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)



#    ....QUERY GENERATOR AGENT....

QUERY_AGENT_PROMPT = """ You are a helpful assistant that can generate search queries for research.
For each query, follow these steps:

1. First, think through and explain:
   - Break down the key aspects that need to be researched
   - Consider potential challenges and how you'll address them
   - Explain your strategy for finding comprehensive information

2. Then generate 3 search queries that:
   - Are specific and focused on retrieving high-quality information
   - Cover different aspects of the topic
   - Will help find relevant and diverse information

Always provide both your thinking process and the generated queries.
"""


class QueryResponse(BaseModel):
    queries: list[str]
    thoughts: str


query_agent = Agent(
    name="Query Generator Agent",
    instructions=QUERY_AGENT_PROMPT,
    output_type=QueryResponse,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash", openai_client=client),
)

#      ........SEARCH AGENT............

class SearchResult(BaseModel):
    title: str
    url: str
    summary: str
    
@function_tool
def url_scrape(url: str) -> str:
    """
    Scrapes a website for it's contents given a url
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator=' ', strip=True)
            
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000] if len(text) > 5000 else text
        except ImportError:
            return response.text[:5000]
    except Exception as e:
        return f"Failed to scrape content from {url}: {str(e)}"

SEARCH_AGENT_PROMPT = (
    "You are a research assistant. Given a URL and its title, you will analyze the content of the URL "
    "and produce a concise summary of the information. The summary must be 2-3 paragraphs."
    "Capture the main points. Write succinctly, no need to have complete sentences or perfect "
    "grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary "
    "itself."
)

search_agent = Agent(
    name="Search Agent",
    instructions=SEARCH_AGENT_PROMPT,
    tools=[url_scrape],
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash", openai_client=client),
)

#      ........SYNTHESIS AGENT............

SYNTHESIS_AGENT_PROMPT = (
    "You are a research report writer. You will receive an original query followed by multiple summaries "
    "of web search results. Your task is to create a comprehensive report that addresses the original query "
    "by combining the information from the search results into a coherent whole. "
    "The report should be well-structured, informative, and directly answer the original query. "
    "Focus on providing actionable insights and practical information. "
    "Aim for up to 5-6 pages with clear sections and a conclusion. "
    "Important: Use markdown formatting with headings and subheadings. Have a table of contents in the beginning of the report that links to each section."
    "Try and include in-text citations to the sources used to create the report with a source list at the end of the report."
)
synthesis_agent = Agent(
    name="Synthesis Agent",
    instructions=SYNTHESIS_AGENT_PROMPT,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash", openai_client=client),
)

#      ........FOLLOW-UP AGENT............

FOLLOW_UP_DECISION_PROMPT = (
    "You are a researcher that decides whether we have enough information to stop "
    "researching or whether we need to generate follow-up queries. "
    "You will be given the original query and summaries of information found so far. "
    
    "IMPORTANT: For simple factual questions (e.g., 'How long do dogs live?', 'What is the height of Mount Everest?'), "
    "if the basic information is already present in the findings, you should NOT request follow-up queries. "
    
    "Complex questions about processes, comparisons, or multifaceted topics may need follow-ups, but simple factual "
    "questions rarely need more than one round of research. "
    
    "If you think we have enough information, return should_follow_up=False. If you think we need to generate follow-up queries, return should_follow_up=True. "
    "If you return True, you will also need to generate 2-3 follow-up queries that address specific gaps in the current findings. "
    "Always provide detailed reasoning for your decision."
)

class FollowUpDecisionResponse(BaseModel):
    should_follow_up: bool
    reasoning: str
    queries: list[str]

follow_up_decision_agent = Agent(
    name="Follow-up Decision Agent",
    instructions=FOLLOW_UP_DECISION_PROMPT,
    output_type=FollowUpDecisionResponse,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash", openai_client=client),
)
