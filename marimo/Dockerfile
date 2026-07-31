FROM ghcr.io/marimo-team/marimo:latest-data

USER root
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

USER appuser

RUN mkdir -p /home/appuser/.config/matplotlib
ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib

COPY marimo.toml /home/appuser/.config/marimo/marimo.toml
