FROM osimis/orthanc:23.2.0

ARG PACKAGE_VERSION
ARG GIT_VERSION

ENV VERBOSE_ENABLED=true
ENV VERBOSE_STARTUP=true
ENV BOA_VERSION=$PACKAGE_VERSION
ENV BOA_GITHASH=$GIT_VERSION

RUN pip3 install celery==5.2.7 unidecode==1.3.6 requests==2.31.0 psycopg2-binary==2.9.9

COPY scripts/*.py /

COPY orthanc.json /etc/orthanc/
