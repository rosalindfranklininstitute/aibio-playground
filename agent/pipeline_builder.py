import inspect
import json
import copy
from dataclasses import dataclass, field
from functools import partial, reduce
from jinja2 import Template
from openai import OpenAI

OLLAMA_BASE_URL = "http://ollama:11434/v1"
MODEL = "gemma4:26b"
TRIAGE_PROMPT = "/app/agent/prompts/triage-prompt.j2"
PIPELINE_PROMPT = "/app/agent/prompts/pipeline-prompt.j2"
EXPLAIN_PROMPT = "/app/agent/prompts/explain-prompt.j2"
MAX_PIPELINE_LENGTH = 10


def get_client() -> OpenAI:
    """Shared Ollama-compatible client reused across agents."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ignored", timeout=120)

def _get_msg(msg, name='content') -> str:
    """Extract named attribute from a mo.ui.chat message or plain dict (testing)"""
    return getattr(msg, name) if hasattr(msg, name) else msg.get(name, '')

def _build_catalogue_json(catalogue: dict) -> str:
    """
    Convert catalogue tool_specs to JSON string, stripping 'image'
    parameter (not relevant for pipeline construction)
    """
    specs = []
    for entry in catalogue.values():
        spec = copy.deepcopy(entry['tool_spec'])
        spec['function']['parameters']['properties'].pop('image', None)
        required = spec['function']['parameters']['required']
        if 'image' in required:
            required.remove('image')
        specs.append(spec)
    return json.dumps(specs, indent=2)

def get_triage_prompt(catalogue: dict) -> str:
    """Template prompt for triage agent from file"""
    with open(TRIAGE_PROMPT, 'r') as f:
        template = Template(f.read())
    return template.render(catalogue_prompt=_build_catalogue_json(catalogue))

def get_pipeline_prompt(catalogue: dict) -> str:
    """Template prompt for pipeline-construction agent from file"""
    with open(PIPELINE_PROMPT, 'r') as f:
        template = Template(f.read())
    return template.render(
        catalogue_prompt=_build_catalogue_json(catalogue),
        max_pipeline_length=MAX_PIPELINE_LENGTH,
    )

def get_explain_prompt(steps_description: str) -> str:
    """Template prompt for pipeline-explanation agent from file"""
    with open(EXPLAIN_PROMPT, 'r') as f:
        template = Template(f.read())
    return template.render(steps_description=steps_description)

# Flag tool called by triage agent to indicate pipeline-construction agent should be called
PROPOSE_PIPELINE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_pipeline",
        "description": (
            "Call this once the user has explicitly agreed that a "
            "pipeline should be built."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

def triage(messages: list, catalogue: dict, client) -> dict:
    """
    Call triage agent responsible for understanding user goals and
    triggering pipeline construction (call propose_pipeline) when appropriate.
    """
    history = [{"role": "system", "content": get_triage_prompt(catalogue)}]
    for msg in messages:
        history.append({"role": _get_msg(msg, 'role'), "content": _get_msg(msg, 'content')})

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=[PROPOSE_PIPELINE_TOOL],
        tool_choice="auto",
        stream=False,
    )

    if not response.choices:
        raise ValueError("LLM returned no choices")

    message = response.choices[0].message
    reasoning = getattr(message, 'reasoning', None)

    if message.tool_calls:
        return {
            "proposed": True,
            "message": None,
            "reasoning": reasoning,
        }
    else:
        return {
            "proposed": False,
            "message": message.content,
            "reasoning": reasoning,
        }

def _strip_code_fences(raw: str) -> str:
    """
    Strip markdown code fences from a string if present.
    Handles ```json ... ``` and ``` ... ``` variants.
    """
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw

def _json_parse_validate(raw: str, catalogue: dict) -> tuple[list, list]:
    """
    Parse raw JSON from LLM and validate steps against catalogue.
    Merge LLM args with catalogue defaults.
    """
    raw = _strip_code_fences(raw)

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], [f"Failed to parse JSON: {e}"]

    if not isinstance(steps, list):
        return [], ["Expected a JSON array of steps."]

    if not steps:
        return [], ["LLM returned an empty pipeline."]

    steps = steps[:MAX_PIPELINE_LENGTH]

    valid, errors = [], []
    for i, step in enumerate(steps):
        name = step.get('name')
        llm_args = step.get('args', {})
        if name not in catalogue:
            errors.append(f"Step {i+1}: unknown function '{name}'")
            continue
        defaults = catalogue[name]['defaults']
        args = {**defaults, **llm_args}
        valid.append({'name': name, 'args': args})

    return valid, errors

def select_pipeline(messages: list, catalogue: dict, client) -> dict:
    """
    Prompt model for an ordered pipeline of 1 or more catalogue functions.
    Uses full conversation history and catalogue for context.
    System prompt ensures strict JSON output, no direct tool calling.
    """
    system_prompt = get_pipeline_prompt(catalogue)

    history = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        history.append({"role": _get_msg(msg, 'role'), "content": _get_msg(msg, 'content')})

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=False,
    )

    if not response.choices:
        raise ValueError("Model returned no choice in pipeline construction.")

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

    error_note = f"\n\n*Validation warnings: {errors}*" if errors else ""
    sources = {
        step['name']: inspect.getsource(catalogue[step['name']]['function'])
        for step in pipeline
    }

    return {
        "selected": True,
        "pipeline": pipeline,
        "message": error_note,
        "reasoning": reasoning,
        "sources": sources,
    }

def explain_pipeline(pipeline: list, catalogue: dict, client, messages: list) -> str:
    """Explain constructed pipeline, using full message history and pipeline as context"""
    steps_description = "\n".join(
        f"{i+1}. {step['name']} (args: {step['args']}): "
        f"{catalogue[step['name']]['metadata']['description']}"
        for i, step in enumerate(pipeline)
    )

    history = []
    for msg in messages:
        history.append({"role": _get_msg(msg, 'role'), "content": _get_msg(msg, 'content')})

    history.append({"role": "user", "content": get_explain_prompt(steps_description)})

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=False,
    )
    if not response.choices:
        raise ValueError("Model returned no choices in explanation call")
    return response.choices[0].message.content

def _build_user_content(user_message: str, image_b64: str = None, image_mime: str = "image/png"):
    """
    Build message content: a plain string, or an OpenAI-style
    multimodal content list if an image is provided.
    """
    if image_b64 is None:
        return user_message
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
        },
        {"type": "text", "text": user_message},
    ]

@dataclass
class ConversationState:
    messages: list = field(default_factory=list)

def run_pipeline(image, pipeline: list, catalogue: dict):
    """
    Apply sequence of catalogue functions to an image.
    Pipeline is an ordered list of {"name": str, "args": dict} steps
    """
    sequence = [
        partial(catalogue[step['name']]['function'], **step['args'])
        for step in pipeline
        if step['name'] in catalogue
    ]
    return reduce(lambda im, fn: fn(im), sequence, image)

def step(state: ConversationState, user_message: str, catalogue: dict, client, image_b64: str = None) -> dict:
    """
    Advance chat conversation by one turn.

    Steps call Triage agent until sufficient information and consent is gathered
    from the user. Then, select_pipeline() is called followed by explain_pipeline().
    If the user does not give consent for pipeline construction, or wants to make
    further changes, the Triage agent is called again.
    """
    new_state = copy.copy(state)
    content = _build_user_content(user_message, image_b64)
    new_state.messages = state.messages + [{"role": "user", "content": content}]

    result = triage(new_state.messages, catalogue, client)

    if not result['proposed']:
        new_state.messages = new_state.messages + [{"role": "assistant", "content": result['message']}]
        return {
            "state": new_state,
            "message": result['message'],
            "pipeline": None,
            "reasoning": result['reasoning'],
        }

    pipeline_result = select_pipeline(new_state.messages, catalogue, client)

    if not pipeline_result['selected']:
        # Feed the failure back into the conversation and continue
        reply = pipeline_result['message']
        new_state.messages = new_state.messages + [{"role": "assistant", "content": reply}]
        return {
            "state": new_state,
            "message": reply,
            "pipeline": None,
            "reasoning": result['reasoning'],
        }

    explanation = explain_pipeline(
        pipeline_result['pipeline'], catalogue, client, new_state.messages
    )
    new_state.messages = new_state.messages + [{"role": "assistant", "content": explanation}]
    return {
        "state": new_state,
        "message": explanation,
        "pipeline": pipeline_result['pipeline'],
        "reasoning": result['reasoning'],
    }

def step_id_to_name(step_id: str):
    """Strip trailing #N identifier to leave pure function name"""
    return step_id.split('#')[0].strip()

def generate_step_id(name: str, existing_ids: list):
    """Generate the next identifier for function name"""
    if name not in existing_ids:
        # Not already in pipeline, use name for identifier
        return name
    n = 2
    while f'{name} #{n}' in existing_ids:
        n += 1
    return f'{name} #{n}'
