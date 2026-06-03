ARG DOCKER_PLATFORM=linux/amd64

FROM --platform=${DOCKER_PLATFORM} nvcr.io/nvidia/pytorch:24.12-py3

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
ENV nnUNet_USE_TRITON=0

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-install-project --extra pacs

COPY weights /app/weights
COPY scripts/*.py /app/
COPY body_organ_analysis /app/body_organ_analysis
RUN uv sync --frozen --extra pacs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN kaleido_get_chrome --path /app/chrome
ENV BROWSER_PATH=/app/chrome/chrome-linux64/chrome

RUN chmod a+rwx -R /app

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
