## Tool Call Reference Examples

### Correct Examples

Example 1 - Simple tool with string argument:
~~~json
{
    "thoughts": ["I need to search for current information."],
    "headline": "Searching the web",
    "tool_name": "search_engine",
    "tool_args": {
        "query": "latest LiteLLM release notes"
    }
}
~~~

Example 2 - Tool with boolean argument (use JSON boolean, not string):
~~~json
{
    "thoughts": ["I'll delegate this coding task to a subordinate agent."],
    "headline": "Calling subordinate",
    "tool_name": "call_subordinate",
    "tool_args": {
        "message": "Fix the bug in utils.py line 42",
        "reset": true
    }
}
~~~

Example 3 - Namespaced tool with multiple arguments:
~~~json
{
    "thoughts": ["I need to read a specific file first."],
    "headline": "Reading file",
    "tool_name": "text_editor:read",
    "tool_args": {
        "path": "/project/src/main.py",
        "line_from": 1,
        "line_to": 60
    }
}
~~~

Example 4 - Tool with empty optional arguments (respond to user):
~~~json
{
    "thoughts": ["I have the answer, I'll respond to the user now."],
    "headline": "Sending response",
    "tool_name": "response",
    "tool_args": {
        "message": "The search results show that version 3.2 was released yesterday."
    }
}
~~~

Example 5 - Code execution tool:
~~~json
{
    "thoughts": ["I need to run a command to check the environment."],
    "headline": "Executing code",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "pip list | grep langchain"
    }
}
~~~

### Common Mistakes to AVOID

Mistake 1 - Boolean as string (WRONG):
~~~json
{
    "tool_name": "call_subordinate",
    "tool_args": { "message": "do something", "reset": "true" }
}
~~~
Correct: use JSON boolean `true` not string `"true"`

Mistake 2 - tool_args as string instead of object (WRONG):
~~~json
{
    "tool_name": "response",
    "tool_args": "Hello user"
}
~~~
Correct: tool_args must always be a JSON object `{}`, never a string

Mistake 3 - Extra text outside the JSON object (WRONG):
```
I will search for that now.
~~~json
{
    "tool_name": "search_engine",
    "tool_args": { "query": "test" }
}
~~~
```
Correct: output ONLY the JSON object wrapped in ~~~json fences, no prose before or after

Mistake 4 - Invented tool name not in available tools list (WRONG):
~~~json
{
    "tool_name": "web_search",
    "tool_args": { "query": "test" }
}
~~~
Correct: use only tool names from the available tools list (e.g., `search_engine` not `web_search`)

Mistake 5 - Missing required field tool_args (WRONG):
~~~json
{
    "thoughts": ["I want to respond"],
    "headline": "Responding",
    "tool_name": "response"
}
~~~
Correct: always include `"tool_args": {}` even if empty
