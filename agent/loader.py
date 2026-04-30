import importlib.util
import os
import sys


CATALOGUE_DIR = os.path.join(os.path.dirname(__file__), '..', 'catalogue')


def load_catalogue() -> dict:
    """
    Load image analysis functions from CATALOGUE_DIR for use
    in agent workflow.

    Note functions are loaded to be callabe, although for now
    the agent will only pass the source code back, not actually
    have it executed in the backend (see tool_selector.py)

    Catalogue entries are .py files containing METADATA
    (translated to a tool_spec for a model in Ollama) and
    a function. The function name must match metadata['name'].

    The returned dictionary is keyed by function name, which
    must be unique across the catalogue.

    Returns
    -------
    catalogue : dict
        Key is name of function in catalogue, value is a
        dictionary with 'metadata', 'function' and 'tool_spec'
        (a trivial derivative of 'metadata').
        The 'function' value is a callable.
    """
    catalogue = {}

    for filename in sorted(os.listdir(CATALOGUE_DIR)):
        # any python file except __init__.py
        if filename.startswith('_') or not filename.endswith('.py'):
            continue

        module_path = os.path.join(CATALOGUE_DIR, filename)
        module_name = os.path.splitext(filename)[0]

        # create a module specification and load it
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'METADATA'):
            print(f'Warning: no METADATA found in {module_path}, skipping.')
            continue

        metadata = module.METADATA

        if not hasattr(module, metadata['name']):
            print(f'Warning: no function "{metadata["name"]}" found in {module_path}, skipping.')
            continue

        fn = getattr(module, metadata['name'])

        if not callable(fn):
            print(f'Warning: "{metadata["name"]}" in {module_path} is not callable, skipping.')
            continue

        catalogue[metadata['name']] = {
            'metadata': metadata,
            'function': fn,
            'tool_spec': metadata_to_ollama_tool(metadata),
        }

        # register only after all checks pass
        sys.modules[module_name] = module

    return catalogue


def metadata_to_ollama_tool(metadata: dict) -> dict:
    """Convert a METADATA dict to an Ollama-compatible tool spec."""
    return {
        "type": "function",
        "function": {
            "name": metadata["name"],
            "description": metadata["description"],
            "parameters": {
                "type": "object",
                "properties": metadata["parameters"],
                "required": metadata["required"],
            }
        }
    }
