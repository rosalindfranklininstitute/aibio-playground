import os
import importlib.util
import inspect

import pytest

CATALOGUE_DIR = os.path.join(os.path.dirname(__file__), '..', 'catalogue')
REQUIRED_METADATA_KEYS = ['name', 'description', 'parameters']
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


def check_function(module_path, module_fn, param_list):
    assert callable(module_fn), f'{module_path}: function is not callable'
    sig = inspect.signature(module_fn)
    params = list(sig.parameters.values())
    assert params, f'{module_path}: function takes no arguments, expected "image" as first argument'
    assert params[0].name == 'image', (
        f'{module_path}: first argument must be "image", got "{params[0].name}"'
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
    check_function(module_path, fn, all_params_list)
