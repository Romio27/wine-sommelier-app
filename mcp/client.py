from dotenv import load_dotenv
import asyncio
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent

load_dotenv()


async def main():
    server_path = os.path.join(os.path.dirname(__file__), "server.py")

    client = MultiServerMCPClient(
        {
            "simple_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Get tools from the server
    tools = await client.get_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    llm = ...
    agent = create_agent(llm, tools)
    r = await agent.ainvoke({
        "messages": [HumanMessage("What is current date and time?")]
    }

    )
    print(r)


if __name__ == "__main__":
    asyncio.run(main())
