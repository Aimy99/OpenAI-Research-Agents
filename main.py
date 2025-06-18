
import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel

# Styling with Rich
from rich.console import Console
from rich.prompt import Prompt


# Openai Imports
from openai import AsyncOpenAI
from agents import set_tracing_disabled

from coordinator import ResearchCoordinator


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

console = Console()


async def main() -> None:
    console.print("[bold cyan]Deep Research Tool[bold cyan]")
    console.print(
        "This tool performs in-depth research on any topic using AI agents.")

    # user Query
    query = Prompt.ask("\n[bold]What would you like to research?[/bold]")
    if not query.strip():
        console.print(
            "[bold red]Error:[/bold red] Please provide a valid query.")
        return
    
    research_coordinator = ResearchCoordinator(query)
    result = await research_coordinator.research()

if __name__ == "__main__":
    asyncio.run(main())
