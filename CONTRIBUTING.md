---
title: BAP: Contributing
description: How to contribute to the Bioimage Analysis Playground
---

# Contributing

If you have suggestions for the Playground app or other features, let us know in the
[app discussion thread](https://github.com/rosalindfranklininstitute/aibio-playground/discussions/5).
Report bugs by creating a [new issue](https://github.com/rosalindfranklininstitute/aibio-playground/issues/new).

We also welcome contributions to the image analysis catalogue, in the form of new functions
or enhancements to existing ones.

## Adding a Catalogue Function
Catalogue functions should be placed in a new file in the `catalogue/` directory
and include a `METADATA` dictionary with `name`, `description`, `parameters`,
`required`, `tags` and `dependencies` entries&mdash;see
[`catalogue/gaussian_blur.py`](catalogue/gaussian_blur.py) for an example. New
dependencies (Python packages) will need to be added to
`marimo/requirements.txt`.  If you are unsure whether a dependency is already
present in the application, you can search the package list in the Marimo
notebook UI, view the list of explicitly installed packages in
`marimo/requirements.txt`, or create an issue asking us to check.

Catalogue functions must accept a dictionary named `image_data` as their first argument.
This dictionary contains:
- `source`: the original image data — not to be modified
- `current`: the image data to be modified by the function
- `info`: a dictionary you can optionally add information to

The modified dictionary should be the function's single return value.

There is no strict linting for Python code by try to follow 
[PEP 8](https://peps.python.org/pep-0008/) style conventions if you can.

## Testing
When you push to GitHub, a workflow automatically checks the syntax and form of
your contribution (e.g., that `METADATA` is well-formed and the function
signature matches the expected interface). This automated check does **not**
verify that your function works correctly: **you are responsible for checking
its functionality** before submitting.

Please test your function against real image data before opening a pull request.

## Submitting a Pull Request
To submit a catalogue function or other change, fork the repository and open a pull request
for review. In your PR description, please include:

- A short summary of what the function does and why it's useful
- A worked example (e.g., sample image, parameters used, and the resulting output) that a
  reviewer can run to verify the function works as intended

This helps reviewers test functionality quickly, since the automated checks only cover
syntax and form.
