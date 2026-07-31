# Bioimage Analysis Playground
<p align="center">
  <img src="/assets/project5.jpg" width=450 />
</p>

## About The Project
This project aims to provide a gateway for newcomers to learn the language of bioimage analysis, 
play with image processing techniques using their own data, and be signposted toward relevant 
tools and communities to continue learning and analysis.

The 'Bioimage Analysis Playground' is a Marmio notebook application that uses
calls to a local LLM (`gemma4:26b`) to help you construct an image anaylsis
pipeline, drawing on a curated library of image analysis functions.

## Deployment
To run the Playground you will need a docker-compose compatible container runtime and an
NVIDIA GPU with sufficient VRAM for the `gemma4:26b` model (>32GB recommended).

Clone and navigate to the repository root, then run:

```bash
docker-compose up -d
```

This builds and runs the Ollama and Marimo container images. By default, Marimo will be
accessible at [http://localhost:8080](http://localhost:8080). Select the `playground.py`
notebook to open the application. Once the notebook has loaded, click the Play icon in the
bottom-right corner to execute all cells, then switch to the app view using the central
square button above the Play icon (`Ctrl+.`).

## Contributing
If you have suggestions for the Playground app or other features, let us know in the
[app discussion thread](https://github.com/rosalindfranklininstitute/aibio-playground/discussions/5).
Report bugs by creating a [new issue](https://github.com/rosalindfranklininstitute/aibio-playground/issues/new).

We also welcome contributions to the image analysis catalogue, in the form of new functions
or enhancements to existing ones.

Catalogue functions should be placed in a new file in the `catalogue/` directory and include
a `METADATA` dictionary with `name`, `description`, `parameters`, `required`, `tags` and
`dependencies` entries — see [`catalogue/gaussian_blur.py`](catalogue/gaussian_blur.py) for
an example. If you are unsure whether a dependency is installed in the Marimo environment,
you can search the package list in the Marimo notebook UI, or create an issue asking us to
check.

Catalogue functions must accept a dictionary named `image_data` as their first argument.
This dictionary contains:
- `source`: the original image data — not to be modified
- `current`: the image data to be modified by the function
- `info`: a dictionary you can optionally add information to

The modified dictionary should be the function's single return value.

To submit a catalogue function, fork the repository and create a pull request for review.

## Funding Statement
This project is supported by AIBIO's [Pilot Project funding](https://aibio.ac.uk/pilot-funding-call/).
