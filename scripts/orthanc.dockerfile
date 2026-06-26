# syntax=docker/dockerfile:1
FROM orthancteam/orthanc:26.4.2

ENV VERBOSE_ENABLED=true
ENV VERBOSE_STARTUP=true

# BUILD_CACHE=1 to reuse uv's and pip's download/build cache across builds
ARG BUILD_CACHE=0

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages \
    $( [ "$BUILD_CACHE" = "1" ] || echo --no-cache-dir ) \
    celery==5.6.3 unidecode==1.4.0 requests==2.34.0 psycopg2-binary==2.9.12

COPY scripts/*.py /

COPY orthanc.json /etc/orthanc/
