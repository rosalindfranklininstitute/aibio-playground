import inspect, json, copy
from openai import OpenAI

OLLAMA_BASE_URL = "http://ollama:11434/v1"
MODEL = "gemma4:26b"
SYSTEM_PROMPT_PATH = "/app/agent/prompts/system-prompt.txt"
MAX_PIPELINE_LENGTH = 10


def load_system_prompt() -> str:
    """Load a default system prompt if one is not provided by file (OUTDATED)"""
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

def _get_msg_content(msg) -> str:
    """Extract content from message object (mo.ui.chat) or dict (testing)"""
    return msg.content if hasattr(msg, 'content') else msg.get('content', '')


def _get_msg_role(msg) -> str:
    """Extract role from message object (mo.ui.chat) or dict (testing)"""
    return msg.role if hasattr(msg, 'role') else msg.get('role', '')


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


def select_tool(messages: list, catalogue: dict):
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
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ignored", timeout=120)
    tools = [entry['tool_spec'] for entry in catalogue.values()]

    history = [{"role": "system", "content": load_system_prompt()}]
    for msg in messages:
        history.append({"role": _get_msg_role(msg), "content": _get_msg_content(msg)})

    print('tools:', tools)
    print('Making API call..')
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=tools,
        tool_choice="auto",
        stream=False,
    )
    print('Response:', response)

    if not response.choices:
        raise ValueError("LLM returned no choices")

    message = response.choices[0].message
    print("message fields:", vars(message))
    reasoning = getattr(message, 'reasoning', None)

    if not message.tool_calls:
        return {
            "selected": False,
            "name": None,
            "args": None,
            "message": message.content,
            "reasoning": reasoning,
            "source": None,
        }

    if len(message.tool_calls) > 1:
        raise ValueError(
            f"LLM returned {len(message.tool_calls)} tool calls, expected 1. "
            f"Names: {[tc.function.name for tc in message.tool_calls]}"
        )

    tool_call = message.tool_calls[0]
    name = tool_call.function.name

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

    fn = catalogue[name]['function']
    source = inspect.getsource(fn)
    description = catalogue[name]['metadata']['description']
    user_query = _get_msg_content(messages[-1])
    explanation = explain_selection(user_query, name, description, client)

    return {
        "selected": True,
        "name": name,
        "args": args,
        "message": f"**Function selected:** `{name}`\n\n{explanation}",
        "reasoning": reasoning,
        "source": source,
    }


def _build_catalogue_prompt(catalogue: dict) -> str:
    """
    Convert catalogue tool_specs into a JSON string for use in
    system prompt (remove 'image' parameter as superfluous to agent)
    """
    specs = []
    for entry in catalogue.values():
        spec = copy.deepcopy(entry['tool_spec'])
        props = spec['function']['parameters']['properties']
        required = spec['function']['parameters']['required']
        props.pop('image', None)
        if 'image' in required: # is a list
            required.remove('image')
        specs.append(spec)
    return json.dumps(specs, indent=2)


def _strip_code_fences(raw: str) -> str:
    """Strip any (json) code fences"""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _get_catalogue_defaults(name: str, catalogue: dict) -> dict:
    """default parameter values for a catalogue func (see loader.py)"""
    return catalogue[name]['defaults']

def _json_parse_validate(raw: str, catalogue: dict) -> tuple[list, list]:
    """
    Parse string from LLM and validate as JSON with string of functions
    from catalogue

    Returns
    -------
    valid_steps : list of {"name": str, "args": dict}
    errors : list of str describing any issues encountered
    """
    raw = _strip_code_fences(raw)

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], [f"LLM returned invalid JSON: {e}"]

    if not isinstance(steps, list):
        return [], ["LLN did not return a JSON array of steps."]

    if not steps:
        return [], ["LLM returned an empty pipeline."]

    steps = steps[:MAX_PIPELINE_LENGTH] # LLM should only return up to MAX_PIPELINE_LENGTH functions anyway

    valid, errors = [], []
    for i, step in enumerate(steps):
        name = step.get('name')
        llm_args = step.get('args', {}) # LLM may not provide args
        if name not in catalogue: # Exact match only (fuzzy instead or second AI to correct?)
            errors.append(f"Step {i+1}: unknown function '{name}'")
            continue
        defaults = _get_catalogue_defaults(name, catalogue)
        args = {**defaults, **llm_args} # LLM args override defaults
        valid.append({'name': name, 'args': args})

    return valid, errors


def _explain_pipeline(pipeline: list, catalogue: dict, client, messages: list) -> str:
    """ Second LLM call to explain pipeline and flag e.g. ordering issues"""

    # include description of each function in pipeline
    steps_description = "\n".join(
        f"{i+1}. {step['name']} (args: {step['args']}): "
        f"{catalogue[step['name']]['metadata']['description']}"
        for i, step in enumerate(pipeline)
    )

    history = []
    # Include chat history with user for context (i.e. need to know their goals)
    for msg in messages:
        history.append({"role": _get_msg_role(msg), "content": _get_msg_content(msg)})

    history.append({
        "role": "user",
        "content": (
            f"The following pipeline of image processing steps was selected:\n"
            f"{steps_description}\n\n"
            f"Briefly explain what each step will do and why this pipeline "
            f"matches the request. If any steps appear to be in an "
            f"unusual or potentially incorrect order, flag this clearly. "
            f"Do not write any code."
        )
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=False,
    )
    if not response.choices:
        raise ValueError("LLM returned no choices in pipeline explanation call")
    return response.choices[0].message.content


def select_pipeline(messages: list, catalogue: dict) -> dict:
    """
    Ask an LLM to suggest an ordered pipeline of of catalogue
    functions appropriate to the user's request.

    Uses a single LLM call with no tool-calling. The catalogue is
    described via serialised tool_spec JSON embedded in the system
    prompt. A second call explains the selected pipeline in natural
    language and flags any ordering concerns.

    Parameters
    ----------
    messages : list
        Conversation history from mo.ui.chat or list of dicts
        (local testing). Each item has a role and content.
    catalogue : dict
        Loaded catalogue from agent.loader.load_catalogue().

    Returns
    -------
    dict with keys:
        'selected'  : bool
        'pipeline'  : list of {"name": str, "args": dict} or None
        'message'   : str - explanation or error description
        'reasoning' : str or None - model thinking if available
        'sources'   : dict of {name: source_code} or None
    """
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ignored", timeout=120)
    catalogue_prompt = _build_catalogue_prompt(catalogue)

    system_prompt = (
        "You are a image processing assistant for microscopy images in biology"
        "You have knowledge of the following image processing functions, "
        "described in JSON:\n\n"
        f"{catalogue_prompt}\n\n"
        "The user will describe what they want to achieve with their image. "
        "It is your job to suggest a series of functions (in order) to apply "
        "to achieve their goal, if possible with the above functions. "
        "Respond ONLY with a valid JSON array of up to "
        f"{MAX_PIPELINE_LENGTH} steps. "
        "Each step must be a JSON object with exactly two keys:\n"
        "  \"name\": the function name (string, must match a name above)\n"
        "  \"args\": an object of parameter values to override defaults "
        "(use {} if all defaults are appropriate)\n"
        "Order steps as they should be applied to the image. "
        "Use only function names from the list above. "
        "If no functions are appropriate, return an empty array: []\n"
        "Do not include any explanation, markdown, or code fences. "
        "Your entire response must be a single JSON array.\n"
        "Example: [{\"name\": \"gaussian_blur\", \"args\": {\"kernel_size\": 5}}, "
        "{\"name\": \"otsu_thresh\", \"args\": {}}]"
    )

    history = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        history.append({"role": _get_msg_role(msg), "content": _get_msg_content(msg)})

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=False,
    )

    if not response.choices:
        raise ValueError("LLM returned no choices")

    message = response.choices[0].message
    reasoning = getattr(message, 'reasoning', None)
    raw = message.content.strip()

    pipeline, errors = _json_parse_validate(raw, catalogue)

    if not pipeline:
        return {
            "selected": False,
            "pipeline": None,
            "message": f"No valid pipeline could be constructed. Errors: {errors}",
            "reasoning": reasoning,
            "sources": None,
        }

    explanation = _explain_pipeline(pipeline, catalogue, client, messages)
    error_note = f"\n\n*Validation warnings: {errors}*" if errors else ""
    sources = {
        step['name']: inspect.getsource(catalogue[step['name']]['function'])
        for step in pipeline
    }

    return {
        "selected": True,
        "pipeline": pipeline,
        "message": f"{explanation}{error_note}",
        "reasoning": reasoning,
        "sources": sources,
    }
