# Gemini API Python SDK Guide

Model: `gemini-3-flash-preview`

## 1. Setup & Initialization

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
```

## 2. Basic Generate (Non-Streaming)

```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What is 2+2?",
)
print(response.text)
```

With system instruction:
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Hello",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
    ),
)
```

## 3. Function Declarations

Two ways to declare functions:

### A. JSON Schema (Manual)
```python
get_weather = {
    "name": "get_weather",
    "description": "Gets weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'London'",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
        },
        "required": ["location"],
    },
}

tools = types.Tool(function_declarations=[get_weather])
```

### B. Python Callable (Auto-Declaration)
```python
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Gets weather for a location.

    Args:
        location: City name, e.g. 'London'
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather data dict
    """
    # Your implementation
    return {"temp": 20, "unit": unit}

# Pass directly as tool - SDK generates declaration
config = types.GenerateContentConfig(tools=[get_weather])
```

## 4. Manual Function Calling Loop

When you need control over function execution:

```python
# Define tools
search_db = {
    "name": "search_db",
    "description": "Search the database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
}

tools = types.Tool(function_declarations=[search_db])
config = types.GenerateContentConfig(
    tools=[tools],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# Initial request
contents = [
    types.Content(role="user", parts=[types.Part.from_text("Find users named John")])
]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=config,
)

# Check for function calls
if response.function_calls:
    for fc in response.function_calls:
        print(f"Function: {fc.name}, Args: {fc.args}")

        # Execute your function
        result = execute_function(fc.name, fc.args)

        # Append model response (contains thought_signature - CRITICAL)
        contents.append(response.candidates[0].content)

        # Append function response
        function_response_part = types.Part.from_function_response(
            name=fc.name,
            response={"result": result},
        )
        contents.append(types.Content(role="user", parts=[function_response_part]))

    # Get final response
    final_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )
    print(final_response.text)
else:
    print(response.text)
```

## 5. Parallel Function Calls

Model can return multiple function calls in one response. Execute them in parallel for efficiency.

### Setup
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

func_a = {
    "name": "get_stock_price",
    "description": "Get stock price",
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}

func_b = {
    "name": "get_company_info",
    "description": "Get company info",
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}

tools = types.Tool(function_declarations=[func_a, func_b])
config = types.GenerateContentConfig(
    tools=[tools],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
```

### Sequential Execution (Simple)
```python
contents = [
    types.Content(role="user", parts=[types.Part.from_text("Get price and info for AAPL")])
]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=config,
)

if response.function_calls:
    contents.append(response.candidates[0].content)

    function_response_parts = []
    for fc in response.function_calls:
        result = execute_function(fc.name, fc.args)
        function_response_parts.append(
            types.Part.from_function_response(name=fc.name, response={"result": result})
        )

    contents.append(types.Content(role="user", parts=function_response_parts))

    final_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )
```

### Parallel Execution (Async)
```python
# Function registry - map names to implementations
FUNCTION_REGISTRY = {
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
    "search_web": search_web,
}

async def execute_function_async(fc_name: str, fc_args: dict) -> dict:
    """Execute a function asynchronously."""
    func = FUNCTION_REGISTRY.get(fc_name)
    if not func:
        return {"error": f"Unknown function: {fc_name}"}

    # If function is async, await it; otherwise run in thread pool
    if asyncio.iscoroutinefunction(func):
        return await func(**fc_args)
    else:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, lambda: func(**fc_args))

async def handle_parallel_function_calls(
    response,
    contents: list,
    config,
) -> str:
    """Handle multiple function calls in parallel."""
    if not response.function_calls:
        return response.text

    # Append model response (preserves thought_signature)
    contents.append(response.candidates[0].content)

    # Execute ALL function calls in parallel
    tasks = [
        execute_function_async(fc.name, fc.args)
        for fc in response.function_calls
    ]
    results = await asyncio.gather(*tasks)

    # Build function response parts (must match order of function calls)
    function_response_parts = [
        types.Part.from_function_response(
            name=fc.name,
            response={"result": result}
        )
        for fc, result in zip(response.function_calls, results)
    ]

    contents.append(types.Content(role="user", parts=function_response_parts))

    # Get next response
    next_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )

    # Recurse if more function calls
    if next_response.function_calls:
        return await handle_parallel_function_calls(next_response, contents, config)

    return next_response.text

# Usage
async def main():
    contents = [
        types.Content(role="user", parts=[types.Part.from_text("Get price and info for AAPL, GOOGL, MSFT")])
    ]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )

    result = await handle_parallel_function_calls(response, contents, config)
    print(result)

asyncio.run(main())
```

### Parallel Execution (ThreadPool - Sync Context)
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def handle_parallel_fc_sync(response, contents: list, config) -> str:
    """Handle parallel function calls in sync context using threads."""
    if not response.function_calls:
        return response.text

    contents.append(response.candidates[0].content)

    # Execute in parallel using ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor(max_workers=len(response.function_calls)) as executor:
        future_to_fc = {
            executor.submit(execute_function, fc.name, fc.args): fc
            for fc in response.function_calls
        }
        for future in as_completed(future_to_fc):
            fc = future_to_fc[future]
            results[id(fc)] = future.result()

    # Build responses in original order
    function_response_parts = [
        types.Part.from_function_response(
            name=fc.name,
            response={"result": results[id(fc)]}
        )
        for fc in response.function_calls
    ]

    contents.append(types.Content(role="user", parts=function_response_parts))

    next_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )

    if next_response.function_calls:
        return handle_parallel_fc_sync(next_response, contents, config)

    return next_response.text
```

## 6. Multi-Turn Conversations

### Using Chat Session (Recommended)
```python
chat = client.chats.create(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
    ),
)

# Conversation
response1 = chat.send_message("My name is Alice")
print(response1.text)

response2 = chat.send_message("What's my name?")
print(response2.text)  # Will remember "Alice"

# Access history
for content in chat.get_history():
    print(f"{content.role}: {content.parts[0].text}")
```

### Manual History Management
```python
history = []

def chat_turn(user_message: str) -> str:
    history.append(
        types.Content(role="user", parts=[types.Part.from_text(user_message)])
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=history,
    )

    # Append full response (preserves any signatures)
    history.append(response.candidates[0].content)
    return response.text
```

## 7. Subagents Pattern

**CRITICAL**: Google Search and function calling CANNOT be used together in the same model call. Use separate subagents.

```python
# Subagent 1: Search Agent (has Google Search, no function calling)
def search_agent(query: str) -> str:
    """Subagent that can search the web."""
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction="Search and summarize. Include source URLs.",
        ),
    )
    return response.text

# Subagent 2: Analysis Agent (has function calling, no search)
def analysis_agent(data: str) -> str:
    """Subagent that can call analysis functions."""
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=data,
        config=types.GenerateContentConfig(
            tools=[analyze_data, generate_report],  # Python callables
        ),
    )
    return response.text

# Main Orchestrator: Uses subagents as tools
def run_orchestrator(task: str) -> str:
    chat = client.chats.create(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            tools=[search_agent, analysis_agent],  # Subagents as tools
            system_instruction="You orchestrate tasks using subagents.",
        ),
    )
    response = chat.send_message(task)
    return response.text
```

### Calling Same Function Multiple Times
Model can call the same function with different args in parallel:
```python
# User: "Get weather for London, Paris, and Tokyo"
# Model returns 3 function calls to get_weather with different locations
for fc in response.function_calls:
    # fc.name = "get_weather" for all three
    # fc.args = {"location": "London"}, {"location": "Paris"}, {"location": "Tokyo"}
```

## 8. Thinking Config

```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Solve this complex problem...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",  # Options: "NONE", "LOW", "MEDIUM", "HIGH"
            include_thoughts=True,  # Include thinking in response
        ),
    ),
)

# Access thinking (if include_thoughts=True)
for part in response.candidates[0].content.parts:
    if hasattr(part, 'thought') and part.thought:
        print(f"Thinking: {part.text}")
    elif part.text:
        print(f"Response: {part.text}")
```

## 9. Google Search Tool

```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What happened in tech news today?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)
print(response.text)

# Access grounding metadata (sources)
if response.candidates[0].grounding_metadata:
    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
        print(f"Source: {chunk.web.uri}")
```

## 10. Thought Signatures (Manual Handling)

**For Gemini 3 models, thought signatures are MANDATORY during function calling.**

The SDK handles this automatically IF you append `response.candidates[0].content` to history. If manipulating history manually:

### Rules:
1. **Parallel FCs**: Signature only on FIRST function call part
2. **Sequential FCs**: Each step has its own signature, must pass ALL back
3. **Missing signature = 400 error**

### Manual Handling (if needed):
```python
# After receiving response with function call
part = response.candidates[0].content.parts[0]
if part.thought_signature:
    # This signature MUST be preserved when sending back
    import base64
    print(f"Signature: {base64.b64encode(part.thought_signature).decode()}")

# CORRECT: Append full content (preserves signature)
contents.append(response.candidates[0].content)

# WRONG: Reconstructing parts manually (loses signature)
# contents.append(types.Content(role="model", parts=[...]))  # DON'T DO THIS
```

### Skip Validation (Emergency Only)
If transferring history from another model:
```python
# Use dummy signature to skip validation
"thought_signature": "skip_thought_signature_validator"
```

## 11. Exception Handling

```python
from google.api_core import exceptions

try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="...",
        config=config,
    )
except exceptions.InvalidArgument as e:
    # 400: Bad request (missing thought_signature, invalid params)
    print(f"Invalid request: {e}")
except exceptions.ResourceExhausted as e:
    # 429: Rate limit
    print(f"Rate limited: {e}")
except exceptions.DeadlineExceeded as e:
    # Timeout
    print(f"Timeout: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## 12. Function Calling Modes

```python
config = types.GenerateContentConfig(
    tools=[tools],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="AUTO",  # Default: model decides
            # mode="ANY",  # Force function call
            # mode="NONE",  # Disable function calling
            # allowed_function_names=["func1", "func2"],  # Restrict to specific functions
        )
    ),
)
```

## Quick Reference

| Task | Method |
|------|--------|
| Basic generation | `client.models.generate_content()` |
| Multi-turn chat | `client.chats.create()` then `chat.send_message()` |
| Function calling (auto) | Pass Python callables to `tools=` |
| Function calling (manual) | Set `automatic_function_calling.disable=True` |
| Parallel FC execution | `asyncio.gather()` or `ThreadPoolExecutor` |
| Google Search | `tools=[types.Tool(google_search=types.GoogleSearch())]` |
| Thinking | `thinking_config=types.ThinkingConfig(thinking_level="HIGH")` |
| System prompt | `system_instruction="..."` in config |
| Export history (chat) | `extract_chat_history(chat)` |
| Export history (contents) | `extract_contents_history(contents)` |
| Track history live | `GeminiHistoryTracker` class |

## Common Patterns

### Orchestrator with Multiple Subagents
```python
def create_orchestrator(subagent_functions: list):
    """Create an orchestrator that can spawn subagents."""
    return client.chats.create(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            tools=subagent_functions,
            system_instruction="""You orchestrate complex tasks.
            Use subagents for specific tasks. Call multiple subagents in parallel when independent.
            Synthesize results into final output.""",
        ),
    )

# Usage
orchestrator = create_orchestrator([search_agent, analysis_agent, writer_agent])
result = orchestrator.send_message("Research and write a report on X")
```

### Loop Until Complete
```python
def run_until_complete(chat, initial_message: str, max_turns: int = 10) -> str:
    """Run conversation until model stops calling functions."""
    response = chat.send_message(initial_message)

    for _ in range(max_turns):
        if not response.function_calls:
            return response.text
        # With automatic_function_calling enabled, SDK handles the loop
        # This is for manual tracking/logging

    return response.text
```

## 13. History Export

Export full conversation history with thinking and output separated, human-readable format.

### History Entry Types
```python
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import json

@dataclass
class HistoryEntry:
    """Single entry in conversation history."""
    timestamp: str
    entry_type: str  # "system" | "user" | "thinking" | "output" | "tool_call" | "tool_response"
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
        }

    def to_readable(self) -> str:
        """Human-readable format."""
        type_labels = {
            "system": "═══ SYSTEM ═══",
            "user": ">>> USER",
            "thinking": "💭 THINKING",
            "output": "<<< MODEL",
            "tool_call": "🔧 TOOL CALL",
            "tool_response": "📥 TOOL RESPONSE",
        }
        label = type_labels.get(self.entry_type, self.entry_type.upper())
        return f"\n{label}\n{'-' * 40}\n{self.content}\n"


@dataclass
class ConversationHistory:
    """Full conversation history with metadata."""
    model: str
    system_instruction: str | None
    entries: list[HistoryEntry] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add(self, entry_type: str, content: str, metadata: dict = None):
        self.entries.append(HistoryEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type=entry_type,
            content=content,
            metadata=metadata or {},
        ))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "system_instruction": self.system_instruction,
            "start_time": self.start_time,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_readable(self) -> str:
        """Human-readable formatted output."""
        lines = [
            "=" * 60,
            f"MODEL: {self.model}",
            f"START: {self.start_time}",
            "=" * 60,
        ]
        if self.system_instruction:
            lines.append(f"\n═══ SYSTEM ═══\n{'-' * 40}\n{self.system_instruction}\n")

        for entry in self.entries:
            lines.append(entry.to_readable())

        return "\n".join(lines)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_json())

    def save_readable(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_readable())
```

### Extract History from Response Parts
```python
def extract_parts_from_response(response) -> dict:
    """Extract thinking, output, and tool calls from a response."""
    result = {
        "thinking": [],
        "output": [],
        "tool_calls": [],
    }

    if not response.candidates:
        return result

    for part in response.candidates[0].content.parts:
        # Check if this is a thinking part
        if hasattr(part, 'thought') and part.thought:
            if hasattr(part, 'text') and part.text:
                result["thinking"].append(part.text)
        # Function call
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            result["tool_calls"].append({
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })
        # Regular output text
        elif hasattr(part, 'text') and part.text:
            result["output"].append(part.text)

    return result
```

### Extract History from Chat Session
```python
def extract_chat_history(
    chat,
    system_instruction: str | None = None,
    model: str = "gemini-3-flash-preview",
) -> ConversationHistory:
    """Extract full history from a chat session with thinking/output separated."""
    history = ConversationHistory(
        model=model,
        system_instruction=system_instruction,
    )

    for content in chat.get_history():
        role = content.role

        for part in content.parts:
            # Thinking content (model only)
            if hasattr(part, 'thought') and part.thought:
                if hasattr(part, 'text') and part.text:
                    history.add("thinking", part.text)

            # Function call
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                args_str = json.dumps(dict(fc.args) if fc.args else {}, indent=2)
                history.add("tool_call", f"{fc.name}({args_str})")

            # Function response
            elif hasattr(part, 'function_response') and part.function_response:
                fr = part.function_response
                resp_str = json.dumps(dict(fr.response) if fr.response else {}, indent=2)
                history.add("tool_response", f"{fr.name} returned:\n{resp_str}")

            # Regular text
            elif hasattr(part, 'text') and part.text:
                entry_type = "user" if role == "user" else "output"
                history.add(entry_type, part.text)

    return history
```

### Extract History from Contents List (generate_content)
```python
def extract_contents_history(
    contents: list,
    system_instruction: str | None = None,
    model: str = "gemini-3-flash-preview",
) -> ConversationHistory:
    """Extract history from contents list with thinking/output separated."""
    history = ConversationHistory(
        model=model,
        system_instruction=system_instruction,
    )

    for content in contents:
        if hasattr(content, 'role'):
            role = content.role
            parts = content.parts if hasattr(content, 'parts') else []
        elif isinstance(content, dict):
            role = content.get('role', 'unknown')
            parts = content.get('parts', [])
        else:
            continue

        for part in parts:
            # Thinking
            if hasattr(part, 'thought') and part.thought:
                if hasattr(part, 'text') and part.text:
                    history.add("thinking", part.text)

            # Function call
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                args_str = json.dumps(dict(fc.args) if fc.args else {}, indent=2)
                history.add("tool_call", f"{fc.name}({args_str})")

            # Function response
            elif hasattr(part, 'function_response') and part.function_response:
                fr = part.function_response
                resp_str = json.dumps(dict(fr.response) if fr.response else {}, indent=2)
                history.add("tool_response", f"{fr.name} returned:\n{resp_str}")

            # Regular text
            elif hasattr(part, 'text') and part.text:
                entry_type = "user" if role == "user" else "output"
                history.add(entry_type, part.text)

            # Dict format
            if isinstance(part, dict):
                if 'text' in part:
                    entry_type = "user" if role == "user" else "output"
                    history.add(entry_type, part['text'])
                if 'functionCall' in part:
                    fc = part['functionCall']
                    args_str = json.dumps(fc.get('args', {}), indent=2)
                    history.add("tool_call", f"{fc.get('name')}({args_str})")
                if 'functionResponse' in part:
                    fr = part['functionResponse']
                    resp_str = json.dumps(fr.get('response', {}), indent=2)
                    history.add("tool_response", f"{fr.get('name')} returned:\n{resp_str}")

    return history
```

### Unified History Tracker Class
```python
class GeminiHistoryTracker:
    """Track conversation history with thinking/output separation."""

    def __init__(self, model: str = "gemini-3-flash-preview", system_instruction: str = None):
        self.history = ConversationHistory(
            model=model,
            system_instruction=system_instruction,
        )

    def log_user_message(self, message: str):
        self.history.add("user", message)

    def log_model_response(self, response):
        """Log model response with thinking/output separated."""
        if not response.candidates:
            return

        for part in response.candidates[0].content.parts:
            # Thinking
            if hasattr(part, 'thought') and part.thought:
                if hasattr(part, 'text') and part.text:
                    self.history.add("thinking", part.text)

            # Function call
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                args_str = json.dumps(dict(fc.args) if fc.args else {}, indent=2)
                self.history.add("tool_call", f"{fc.name}({args_str})")

            # Regular output
            elif hasattr(part, 'text') and part.text:
                self.history.add("output", part.text)

    def log_tool_response(self, function_name: str, response: dict):
        resp_str = json.dumps(response, indent=2)
        self.history.add("tool_response", f"{function_name} returned:\n{resp_str}")

    def to_json(self) -> str:
        return self.history.to_json()

    def to_readable(self) -> str:
        return self.history.to_readable()

    def save(self, filepath: str):
        self.history.save(filepath)

    def save_readable(self, filepath: str):
        self.history.save_readable(filepath)
```

### Usage Example
```python
tracker = GeminiHistoryTracker(
    model="gemini-3-flash-preview",
    system_instruction="You are a helpful weather assistant.",
)

# User message
tracker.log_user_message("What's the weather in London and Paris?")

# Model response (with thinking enabled)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in London and Paris?",
    config=types.GenerateContentConfig(
        tools=[get_weather],
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
            include_thoughts=True,
        ),
    ),
)
tracker.log_model_response(response)

# Tool responses
if response.function_calls:
    for fc in response.function_calls:
        result = execute_function(fc.name, fc.args)
        tracker.log_tool_response(fc.name, result)

# Save both formats
tracker.save("history.json")
tracker.save_readable("history.txt")

# Print readable
print(tracker.to_readable())
```

### Human-Readable Output Example
```
============================================================
MODEL: gemini-3-flash-preview
START: 2024-01-15T10:30:00.000000
============================================================

═══ SYSTEM ═══
----------------------------------------
You are a helpful weather assistant.

>>> USER
----------------------------------------
What's the weather in London and Paris?

💭 THINKING
----------------------------------------
The user wants weather for two cities. I should call
get_weather for both London and Paris. These are
independent queries so I can call them in parallel.

🔧 TOOL CALL
----------------------------------------
get_weather({
  "location": "London"
})

🔧 TOOL CALL
----------------------------------------
get_weather({
  "location": "Paris"
})

📥 TOOL RESPONSE
----------------------------------------
get_weather returned:
{
  "temp": 15,
  "unit": "celsius",
  "condition": "cloudy"
}

📥 TOOL RESPONSE
----------------------------------------
get_weather returned:
{
  "temp": 18,
  "unit": "celsius",
  "condition": "sunny"
}

<<< MODEL
----------------------------------------
The weather in London is 15°C and cloudy.
In Paris, it's 18°C and sunny.
```

### JSON Output Example
```json
{
  "model": "gemini-3-flash-preview",
  "system_instruction": "You are a helpful weather assistant.",
  "start_time": "2024-01-15T10:30:00.000000",
  "entries": [
    {
      "timestamp": "2024-01-15T10:30:01.000000",
      "type": "user",
      "content": "What's the weather in London and Paris?",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:02.000000",
      "type": "thinking",
      "content": "The user wants weather for two cities. I should call get_weather for both...",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:02.000000",
      "type": "tool_call",
      "content": "get_weather({\"location\": \"London\"})",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:02.000000",
      "type": "tool_call",
      "content": "get_weather({\"location\": \"Paris\"})",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:03.000000",
      "type": "tool_response",
      "content": "get_weather returned:\n{\"temp\": 15, \"unit\": \"celsius\", \"condition\": \"cloudy\"}",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:03.000000",
      "type": "tool_response",
      "content": "get_weather returned:\n{\"temp\": 18, \"unit\": \"celsius\", \"condition\": \"sunny\"}",
      "metadata": {}
    },
    {
      "timestamp": "2024-01-15T10:30:04.000000",
      "type": "output",
      "content": "The weather in London is 15°C and cloudy. In Paris, it's 18°C and sunny.",
      "metadata": {}
    }
  ]
}
```
