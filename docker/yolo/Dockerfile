FROM mambaorg/micromamba:1.5.7-bookworm-slim

COPY environment.yml /tmp
RUN micromamba install -f /tmp/environment.yml -n base -y && micromamba clean -ay && /opt/conda/bin/pip cache purge
