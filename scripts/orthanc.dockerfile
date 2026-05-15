FROM orthancteam/orthanc:26.4.2

ARG PACKAGE_VERSION
ARG GIT_VERSION

ENV VERBOSE_ENABLED=true
ENV VERBOSE_STARTUP=true
ENV BOA_VERSION=$PACKAGE_VERSION
ENV BOA_GITHASH=$GIT_VERSION

RUN pip3 install --break-system-packages \
    celery==5.6.3 unidecode==1.4.0 requests==2.34.0 psycopg2-binary==2.9.12

COPY scripts/*.py /

COPY orthanc.json /etc/orthanc/
