import json
from importlib import import_module
from typing import Any

from ..config import AtsiiitsiinConfig
from .tools import tool_add_note, tool_add_tags, tool_search

# Type for messages in the conversation memory
MessageDict = dict[str, Any]


def _completion_call(**kwargs: Any) -> Any:
    """Deferred import wrapper to avoid hard dependency during type checking."""
    completion = import_module("litellm").completion
    return completion(**kwargs)

SYSTEM_PROMPT = (
    "You are Atsiitsʼiin, a second-brain assistant that helps users capture, "
    "organize, and retrieve their thoughts and ideas.\n\n"
    "You have access to tools that let you:\n"
    "- Store notes in memory (add_note): Use when the user wants to remember or save something\n"
    "- Search through stored memories (search_memory): Use when the user asks questions or wants to recall information\n"
    "- Classify notes with tags (add_tags): Use when you want to (re)generate topic tags for an existing note\n\n"
    "Workflow:\n"
    "1. When storing: If the user says 'I want to remember X' or similar, extract a short title (e.g., 'Meeting notes', 'Note about X') and use the user's full statement as content\n"
    "2. When querying: Use search_memory to find relevant chunks, then read the TEXT field from results and synthesize an answer\n"
    "3. When tagging: Call add_tags if the user requests classifications, or after saving a note if additional refinement is needed\n"
    "4. Always cite retrieved chunks when answering - quote or paraphrase the TEXT content you found\n\n"
    "The search_memory tool returns a JSON array. Each item has a 'TEXT' field containing the actual chunk content - use this to answer questions. "
    "Other fields like 'NOTE_ID' and 'CHUNK_INDEX' are for reference."
)

# Tool function registry
TOOL_FUNCTIONS = {
    "add_note": tool_add_note,
    "search_memory": tool_search,
    "add_tags": tool_add_tags,
}

# Tool definitions for the LLM (OpenAI function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Store a note in memory. Use this when the user wants to remember something, save information, or capture a thought. Extract a concise title from the user's message and use their full statement as the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A brief title or summary of what's being remembered",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content of the note to store",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier (defaults to 'user-1')",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source of the note (defaults to 'manual')",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search through stored memories using semantic search. Use this when the user asks questions, wants to recall information, or needs context from past notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - what to look for in memory",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (defaults to 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_tags",
            "description": "Classify an existing note with descriptive tags. Use this to regenerate tags or create them when the user requests categorization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Identifier of the note to classify",
                    },
                    "max_tags": {
                        "type": "integer",
                        "description": "Maximum number of tags to request (defaults to 5)",
                        "default": 5,
                    },
                    "source": {
                        "type": "string",
                        "description": "Origin of the tagging request (defaults to 'llm_suggestion')",
                        "default": "llm_suggestion",
                    },
                },
                "required": ["note_id"],
            },
        },
    },
]


class AtsiiitsiinAgent:
    def __init__(
        self,
        cfg: AtsiiitsiinConfig,
        max_iterations: int = 10,
    ):
        self.cfg = cfg
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        """
        Run the agent with natural language input.

        The agent uses LLM function calling to decide when to store or retrieve memories.
        """
        iterations = 0
        memory: list[MessageDict] = [{"role": "user", "content": user_message}]

        while iterations < self.max_iterations:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory

            try:
                response: Any = _completion_call(
                    model=self.cfg.llm_model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=self.cfg.llm_max_tokens,
                    temperature=self.cfg.llm_temperature,
                    stream=False,
                )

                message = response.choices[0].message

                # Handle tool calls
                if message.tool_calls:
                    # Add assistant message with tool calls to memory
                    memory.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in message.tool_calls
                            ],
                        }
                    )

                    # Execute all tool calls
                    tool_results = []
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        if tool_name in TOOL_FUNCTIONS:
                            try:
                                result = TOOL_FUNCTIONS[tool_name](
                                    tool_args, self.cfg
                                )
                                tool_results.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps(result),
                                    }
                                )
                            except Exception as e:  # noqa: BLE001
                                tool_results.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps(
                                            {"error": f"Error: {str(e)}"}
                                        ),
                                    }
                                )
                        else:
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps(
                                        {"error": f"Unknown tool: {tool_name}"}
                                    ),
                                }
                            )

                    memory.extend(tool_results)
                    iterations += 1
                    continue

                # No tool calls - return the response
                if message.content:
                    return message.content

                # If no content and no tool calls, something went wrong
                return "I'm not sure how to respond to that."

            except Exception as e:  # noqa: BLE001
                return f"Error: {str(e)}"

        # Max iterations reached
        return (
            f"I reached the maximum number of iterations ({self.max_iterations}). "
            "Please try rephrasing your request."
        )
