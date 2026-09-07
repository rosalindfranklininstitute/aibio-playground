---
title: BAP: Deployment
description: How to deploy the Bioimage Analysis Playground
---

# Deployment

## Architecture
The Bioimage Analysis Playground has two main components:

1. A **Marimo server**, comprising backend Python code (e.g., catalogue functions), and a
   frontend notebook file.
2. An **Ollama server**, running Google DeepMind's
   [gemma4:26b](https://ollama.com/library/gemma4:26b) model, which the Marimo server calls
   to interpret user requests and help construct the analysis pipeline.

These two components are deployed as containers in the Docker project defined by
`docker-compose.yml`.

## Requirements
You will need a Docker Compose-compatible container runtime and an NVIDIA GPU with
sufficient VRAM for the [gemma4:26b](https://ollama.com/library/gemma4:26b) model (>32GB
recommended). A web browser is required to view and interact with the playground.

### Other AI Providers
The playground currently requires an LLM served locally via Ollama. Support for models from
web providers such as OpenAI or Anthropic could be considered for a future release.
Let us know if you'd like this feature, or even would like to contribute a PR for it!

## Running the Playground
First, clone the repository and navigate to its base directory:

```bash
git clone git@github.com:rosalindfranklininstitute/aibio-playground.git
cd aibio-playground
```

Then start the playground with:

```bash
docker-compose up -d
```

This builds and runs the Ollama and Marimo container images. By default, Marimo is
accessible at [http://localhost:8080](http://localhost:8080)&mdash;edit the `ports` specification
under `marimo` in `docker-compose.yml` if you need to change this.

Once running, select the `playground.py` notebook to open the application. When the notebook
has loaded, click the Play icon in the bottom-right corner to execute all cells, then switch
to the app view using the central square button above the Play icon (`Ctrl+.`).

### Local Data
Volume mounts are configured for the `/data/marimo` and `notebooks` directories in the
repository. If you have image data elsewhere that you want Marimo to access, add a mount
point under `volumes` for the `marimo` service in `docker-compose.yml`, then restart the
project.
