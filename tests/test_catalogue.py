import os
import importlib.util, importlib.metadata
import inspect
import numpy as np

import pytest

CATALOGUE_DIR = os.path.join(os.path.dirname(__file__), '..', 'catalogue')
REQUIRED_METADATA_KEYS = ['name', 'description', 'parameters', 'required', 'tags', 'dependencies']
REQUIRED_PARAMETER_KEYS = ['type', 'description']


def check_metadata(module_path, module_metadata):
    for KEY in REQUIRED_METADATA_KEYS:
        assert KEY in module_metadata, f'{module_path}: METADATA missing {KEY}'
    all_params_dict = module_metadata['parameters']
    all_params_list = []
    for param, param_dict in all_params_dict.items():
        all_params_list.append(param)
        for KEY in REQUIRED_PARAMETER_KEYS:
            assert KEY in param_dict, f'{module_path}: METADATA parameter {param} missing {KEY}'
    return all_params_list


def check_function_signature(module_path, module_fn, param_list):
    assert callable(module_fn), f'{module_path}: function is not callable'
    sig = inspect.signature(module_fn)
    params = list(sig.parameters.values())
    assert params, f'{module_path}: function takes no arguments, expected "image_data" as first argument'
    assert params[0].name == 'image_data', (
        f'{module_path}: first argument must be "image_data", got "{params[0].name}"'
    )
    sig_names = {p.name for p in params}
    assert sig_names == set(param_list), (
        f'{module_path}: METADATA parameters {sorted(param_list)} '
        f'do not match function arguments {sorted(sig_names)}'
    )
    for p in params[1:]:
        assert p.default is not inspect.Parameter.empty, (
            f'{module_path}: parameter "{p.name}" has no default value'
        )

def check_dependencies(module_path, fn_dependencies):
    for dep in fn_dependencies:
        try:
            importlib.metadata.distribution(dep)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(
                f'{module_path}: dependency "{dep}" listed in METADATA is not installed. '
                f'Add it to marimo/requirements.txt'
            )

def check_function_call(module_fn, image_data):
    original_source = image_data["source"].copy()
    fn_name = module_fn.__name__

    try:
        result = module_fn(image_data)
    except (ModuleNotFoundError, ImportError) as e:
        pytest.fail(
            f'{fn_name}: missing dependency ({e}). '
            f'Ensure is listed in METADATA["dependencies"] and added to marimo/requirements.txt'
        )

    assert isinstance(result, dict), f'{fn_name} must return a dictionary'
    required = {"source", "current", "info"}
    assert required <= result.keys(), f'{fn_name} must have keys "source", "current" and "info"'
    assert np.array_equal(result["source"], original_source), f'{fn_name} should not modify image_data["source"]'
    #assert isinstance(result["current"], np.ndarray), f'{fn_name} must return numpy data array in "current"'

def _catalogue_files():
    return sorted(
        f for f in os.listdir(CATALOGUE_DIR)
        if not f.startswith('_') and f.endswith('.py')
    )


def _load_catalogue_module(fname):
    module_path = os.path.join(CATALOGUE_DIR, fname)
    module_name = os.path.splitext(fname)[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


@pytest.mark.parametrize('fname', _catalogue_files())
def test_catalogue_file(fname):
    module, module_path = _load_catalogue_module(fname)

    assert hasattr(module, 'METADATA'), f'{module_path} has no METADATA dictionary'
    metadata = module.METADATA
    all_params_list = check_metadata(module_path, metadata)

    assert hasattr(module, metadata['name']), f'{module_path} has no function "{metadata["name"]}"'
    fn = getattr(module, metadata['name'])
    check_function_signature(module_path, fn, all_params_list)

    check_dependencies(module_path, metadata['dependencies'])

    # Construct small test input

    # Construct small test input
    if metadata['name'] == 'watershed':
        source = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255
    elif metadata['name'] == 'label_regions' or metadata['name'] == 'label_classes':
        source = np.random.choice(np.linspace(start=0, stop=255, num=4),(100,100)).astype(np.uint8)
    else:
        source = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    image_data = {"source": source.copy(), "current": source.copy(), "info": {}}
    check_function_call(fn, image_data)
