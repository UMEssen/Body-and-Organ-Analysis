FROM python:3.9 as poetry2requirements
COPY pyproject.toml poetry.lock /
ENV POETRY_HOME=/etc/poetry
RUN pip3 install poetry==1.3.2
RUN python3 -m poetry export -E triton --without-hashes -f requirements.txt > /Requirements.txt

FROM nvcr.io/nvidia/pytorch:22.12-py3

# Install app dependencies
COPY --from=poetry2requirements /Requirements.txt /tmp
RUN apt-get -y update; DEBIAN_FRONTEND=noninteractive apt-get -y install \
    curl ffmpeg libsm6 libxext6 libpangocairo-1.0-0 dcmtk xvfb

WORKDIR /app

ENV TOTALSEG_WEIGHTS_PATH="/app/weights"
ENV MPLCONFIGDIR = "/app/configs"
ENV nnUNet_USE_TRITON=0

COPY pyproject.toml README.md /app/
COPY body_organ_analyzer /app/body_organ_analyzer

RUN chmod a+rwx -R /app && \
    pip3 install -U pip && \
    pip3 install -r /tmp/Requirements.txt && \
    rm /tmp/Requirements.txt && \
    pip install pyradiomics && \
    pip install .
