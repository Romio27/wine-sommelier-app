from dotenv import load_dotenv
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("employee-service-mcp")


@mcp.tool(description="Echoes the input text back to the caller.")
def echo(text: Annotated[str, "Text to repeat"]) -> str:
    return text


@mcp.tool()
def time() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
