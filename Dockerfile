FROM mambaorg/micromamba:1.5.7-bookworm-slim

COPY stack_sample_images.py send_birb_summary.py send_email.py capture.py email-and-upload.sh /usr/local/bin
COPY environment.yml /tmp
RUN micromamba install -f /tmp/environment.yml -n base -y && micromamba clean -ay && /opt/conda/bin/pip cache purge
