import inspect, json
from openai import OpenAI

OLLAMA_BASE_URL = "http://ollama:11434/v1"
MODEL = "gemma4:26b"
SYSTEM_PROMPT_PATH = "/app/agent/prompts/system-prompt.txt"


def load_system_prompt() -> str:
    """Load a default system prompt if one is not provided by file"""
    try:
        with open(SYSTEM_PROMPT_PATH, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return (
            "You are an image analysis assistant. "
            "You have access to a catalogue of vetted image processing functions. "
            "When a user asks a question, select the most appropriate function from "
            "the catalogue using the tools provided. Select at most one function. "
            "If no function is appropriate, say so clearly and suggest where the "
            "user might find help. "
            "Never write new code."
        )


def explain_selection(user_query, name, description, client):
    """
    Second LLM call without tools to get a natural language explanation
    of why the selected function matches the user's request.

    Parameters
    ----------
    user_query : str
        The user's original message.
    name : str
        Name of the selected function.
    description : str
        Description from the function METADATA.
    client : OpenAI
        Ollama OpenAI-compatible client.

    Returns
    -------
    str
        Natural language explanation.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"The user asked: \"{user_query}\"\n"
                    f"The function \"{name}\" was selected.\n"
                    f"Description: \"{description}\"\n\n"
                    f"Briefly explain what this function will do to their image "
                    f"and why it matches their request. Do not write any code."
                )
            }
        ],
        stream=False,
    )
    if not response.choices:
        raise ValueError("LLM returned no choices")
    return response.choices[0].message.content


def select_tool(messages: list, catalogue: dict) -> dict:
    """
    Send conversation history and tool specs to a LLM,
    and return the result in a dict.

    Parameters
    ----------
    messages : list
        Full conversation history from mo.ui.chat, each item having
        a role and content. The last message is the latest user
        prompt.
    catalogue : dict
        Loaded catalogue from agent.loader.load_catalogue().

    N.B. For now the LLM/app does not call the function directly,
    only passes the source code around (having fn a callable is
    convenient for this still)

    Returns
    -------
    dict with keys:
        'selected'  : bool - whether the model selects a tool (function)
        'name'      : str or None - name of selected function
        'args'      : dict or None - parameters for function selected
        'message'   : str - LLM response or confirmation of function selected
        'reasoning' : str or None - model thinking
        'source'    : str or None - code for the function selected
    """
    # Create Ollama client (no API key required as local)
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ignored", timeout=120)
    # Get Ollama compatible tool specs for each catalogue function
    tools = [entry['tool_spec'] for entry in catalogue.values()]

    # Build message history for LLM, prepending system prompt
    history = [{"role": "system", "content": load_system_prompt()}]
    # mo.ui.chat uses ChatMessages with role and content attributes
    for msg in messages:
        try:
            role = msg.role
            content = msg.content
        except AttributeError:
            # TESTING - allow dict
            role = msg.get("role")
            content = msg.get("content")
        history.append({"role": role, "content": content})

    # Get (next) response from model
    print('tools:', tools)
    print('Making API call..')
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=tools,
        tool_choice="auto",  # LLM can use a tool, but doesn't have to
        stream=False,  # No streaming (not simple with tool flow)
    )
    print('Response:', response)

    if not response.choices:
        raise ValueError("LLM returned no choices")

    # In general can query model for n>=1 responses, meaning
    # 'choices' is a list (n=1 in above create() call)
    message = response.choices[0].message
    print("message fields:", vars(message))
    reasoning = getattr(message, 'reasoning', None)

    # No tool selected
    if not message.tool_calls:
        return {
            "selected": False,
            "name": None,
            "args": None,
            "message": message.content,
            "reasoning": reasoning,
            "source": None,
        }

    # Don't allow parallel tool calling (for now)
    if len(message.tool_calls) > 1:
        raise ValueError(
            f"LLM returned {len(message.tool_calls)} tool calls, expected 1. "
            f"Names: {[tc.function.name for tc in message.tool_calls]}"
        )

    tool_call = message.tool_calls[0]
    name = tool_call.function.name

    # Return an informative message if LLM tries to call a function
    # that does not exist - later, we could try re-routing this to
    # another call
    if name not in catalogue:
        return {
            "selected": False,
            "name": name,
            "args": None,
            "message": (
                f"The agent tried to find function `{name}` in the catalogue "
                f"but it does not exist. This may indicate the model hallucinated "
                f"a function name."
            ),
            "reasoning": reasoning,
            "source": None,
        }

    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    # Get source code of callable function
    fn = catalogue[name]['function']
    source = inspect.getsource(fn)
    description = catalogue[name]['metadata']['description']

    # Get last user message for explanation context
    user_query = messages[-1].content if hasattr(messages[-1], 'content') else messages[-1].get('content', '')

    # Second LLM call for natural language explanation (no tools)
    explanation = explain_selection(user_query, name, description, client)

    return {
        "selected": True,
        "name": name,
        "args": args,
        "message": f"**Function selected:** `{name}`\n\n{explanation}",
        "reasoning": reasoning,
        "source": source,
    }
