"""Standalone MCP Server with demo tools.

The LLM Gateway connects to this as an MCP client, discovers the tools,
and lets the LLM decide when to call them during an agentic loop.

Run: python mcp_server.py (stdio transport)
"""
from __future__ import annotations
import json
import math
import datetime
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo_tools")

# ── In-memory notes store ──────────────────────────────────────────────────

_notes: dict[str, str] = {}


@mcp.tool(name="add")
def add(a: float, b: float) -> str:
    """Add two numbers and return the sum.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    result = a + b
    return json.dumps({"result": result})


@mcp.tool(name="subtract")
def subtract(a: float, b: float) -> str:
    """Subtract b from a.

    Args:
        a: Number to subtract from.
        b: Number to subtract.

    Returns:
        The difference a - b.
    """
    result = a - b
    return json.dumps({"result": result})


@mcp.tool(name="multiply")
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The product of a and b.
    """
    result = a * b
    return json.dumps({"result": result})


@mcp.tool(name="divide")
def divide(a: float, b: float) -> str:
    """Divide a by b.

    Args:
        a: Numerator.
        b: Denominator (must not be zero).

    Returns:
        The quotient a / b.
    """
    if b == 0:
        return json.dumps({"error": "Division by zero"})
    result = a / b
    return json.dumps({"result": result})


@mcp.tool(name="power")
def power(base: float, exponent: float) -> str:
    """Raise base to the power of exponent.

    Args:
        base: The base number.
        exponent: The exponent.

    Returns:
        base raised to the power of exponent.
    """
    result = math.pow(base, exponent)
    return json.dumps({"result": result})


@mcp.tool(name="sqrt")
def sqrt(number: float) -> str:
    """Calculate the square root of a number.

    Args:
        number: The number (must be non-negative).

    Returns:
        The square root.
    """
    if number < 0:
        return json.dumps({"error": "Cannot take square root of a negative number"})
    result = math.sqrt(number)
    return json.dumps({"result": result})


@mcp.tool(name="save_note")
def save_note(title: str, content: str) -> str:
    """Save a note with a title and content.

    Args:
        title: The title/key for the note.
        content: The text content of the note.

    Returns:
        Confirmation that the note was saved.
    """
    _notes[title] = content
    return json.dumps({"status": "saved", "title": title, "total_notes": len(_notes)})


@mcp.tool(name="read_note")
def read_note(title: str) -> str:
    """Read a note by its title.

    Args:
        title: The title/key of the note to read.

    Returns:
        The note content, or an error if not found.
    """
    if title in _notes:
        return json.dumps({"title": title, "content": _notes[title]})
    return json.dumps({"error": f"Note '{title}' not found", "available_notes": list(_notes.keys())})


@mcp.tool(name="list_notes")
def list_notes() -> str:
    """List all saved note titles.

    Returns:
        A list of all note titles.
    """
    return json.dumps({"notes": list(_notes.keys()), "count": len(_notes)})


@mcp.tool(name="delete_note")
def delete_note(title: str) -> str:
    """Delete a note by title.

    Args:
        title: The title of the note to delete.

    Returns:
        Confirmation or error if note not found.
    """
    if title in _notes:
        del _notes[title]
        return json.dumps({"status": "deleted", "title": title})
    return json.dumps({"error": f"Note '{title}' not found"})


@mcp.tool(name="get_current_time")
def get_current_time(timezone: Optional[str] = None) -> str:
    """Get the current date and time.

    Args:
        timezone: Optional timezone name (currently returns local time regardless).

    Returns:
        Current date and time.
    """
    now = datetime.datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "iso": now.isoformat(),
    })


@mcp.tool(name="string_length")
def string_length(text: str) -> str:
    """Count the number of characters and words in a string.

    Args:
        text: The string to measure.

    Returns:
        Character count and word count.
    """
    return json.dumps({
        "characters": len(text),
        "words": len(text.split()),
    })


@mcp.tool(name="convert_temperature")
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: The temperature value to convert.
        from_unit: Source unit — 'C', 'F', or 'K'.
        to_unit: Target unit — 'C', 'F', or 'K'.

    Returns:
        The converted temperature value.
    """
    from_u = from_unit.upper()
    to_u = to_unit.upper()

    if from_u == to_u:
        return json.dumps({"result": value, "unit": to_u})

    celsius = value
    if from_u == "F":
        celsius = (value - 32) * 5 / 9
    elif from_u == "K":
        celsius = value - 273.15

    if to_u == "C":
        result = celsius
    elif to_u == "F":
        result = celsius * 9 / 5 + 32
    elif to_u == "K":
        result = celsius + 273.15
    else:
        return json.dumps({"error": f"Unknown unit '{to_unit}'. Use C, F, or K."})

    return json.dumps({"result": round(result, 2), "unit": to_u})


if __name__ == "__main__":
    mcp.run()
