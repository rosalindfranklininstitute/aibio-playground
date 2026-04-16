FROM ghcr.io/marimo-team/marimo:latest-data

USER root

RUN pip install --no-cache-dir \
    matplotlib \
    seaborn \
    pillow \
    opencv-python-headless==4.12.0.88 \
    openai \
    chromadb

USER appuser
