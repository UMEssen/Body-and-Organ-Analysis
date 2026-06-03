ARG DOCKER_PLATFORM=linux/amd64

FROM --platform=${DOCKER_PLATFORM} python:3.12-slim-bookworm

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        curl ffmpeg libsm6 libxext6 libpangocairo-1.0-0 dcmtk xvfb libjemalloc2 \
        gosu libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libdrm2 libasound2t64 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

ENV TOTALSEG_WEIGHTS_PATH=/app/weights
ENV MPLCONFIGDIR=/app/configs
ENV nnUNet_USE_TRITON=1

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-install-project --extra triton --extra pacs

COPY weights /app/weights
COPY scripts/*.py /app/
COPY body_organ_analysis /app/body_organ_analysis
RUN uv sync --frozen --extra triton --extra pacs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN kaleido_get_chrome --path /app/chrome
ENV BROWSER_PATH=/app/chrome/chrome-linux64/chrome

# Only the dirs written at runtime by the non-root user need to be writable.
# A recursive chmod over all of /app would rewrite the multi-GB venv/CUDA tree
# into one huge layer (and mark everything world-writable).
RUN mkdir -p /app/weights /app/configs && chmod -R a+rwX /app/weights /app/configs

COPY --chmod=0755 scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
