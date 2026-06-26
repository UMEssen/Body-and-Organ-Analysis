# syntax=docker/dockerfile:1
ARG DOCKER_PLATFORM=linux/amd64

FROM --platform=${DOCKER_PLATFORM} nvcr.io/nvidia/pytorch:24.12-py3

# BUILD_CACHE=1 to reuse uv's and pip's download/build cache across builds
ARG BUILD_CACHE=0

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        curl ffmpeg libsm6 libxext6 libpangocairo-1.0-0 xvfb libjemalloc2 \
        gosu libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libdrm2 libasound2t64 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra pacs $( [ "$BUILD_CACHE" = "1" ] || echo --no-cache )

COPY scripts/*.py /app/
COPY body_organ_analysis /app/body_organ_analysis
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra pacs $( [ "$BUILD_CACHE" = "1" ] || echo --no-cache )

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/weights && \
    python /app/download_weights.py && \
    chmod -R a+rwX /app/weights

RUN mkdir -p /app/configs && chmod -R a+rwX /app/configs

COPY --chmod=0755 scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
