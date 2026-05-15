FROM nvcr.io/nvidia/pytorch:24.12-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        curl ffmpeg libsm6 libxext6 libpangocairo-1.0-0 dcmtk xvfb libjemalloc2 && \
    rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

ARG PACKAGE_VERSION
ARG GIT_VERSION
ENV BOA_VERSION=$PACKAGE_VERSION
ENV BOA_GITHASH=$GIT_VERSION
ENV TOTALSEG_WEIGHTS_PATH=/app/weights
ENV MPLCONFIGDIR=/app/configs
ENV nnUNet_USE_TRITON=0

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-install-project

COPY weights /app/weights
COPY body_organ_analysis /app/body_organ_analysis
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

RUN chmod a+rwx -R /app
