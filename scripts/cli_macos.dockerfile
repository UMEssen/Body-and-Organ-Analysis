ARG DOCKER_PLATFORM=linux/arm64

FROM python:3.12-slim-bookworm

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        curl ffmpeg libsm6 libxext6 libpangocairo-1.0-0 dcmtk xvfb libjemalloc2 \
        gosu chromium build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

# Bare soname (no path) so the dynamic loader resolves the correct multiarch
# location of libjemalloc on whichever architecture this image is built for.
ENV LD_PRELOAD=libjemalloc.so.2

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

COPY body_organ_analysis /app/body_organ_analysis
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Chrome-for-Testing (what `kaleido_get_chrome` downloads) ships no Linux arm64
# build, so use the distro's native chromium for plotly/kaleido image export.
# Only used when PDF generation is enabled; the macOS CI run sets BCA_NO_PDF=1.
ENV BROWSER_PATH=/usr/bin/chromium

# Only the dirs written at runtime by the non-root user need to be writable.
RUN mkdir -p /app/weights /app/configs && chmod -R a+rwX /app/weights /app/configs

COPY --chmod=0755 scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
