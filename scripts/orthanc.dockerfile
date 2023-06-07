FROM osimis/orthanc:23.2.0

ENV VERBOSE_ENABLED=true
ENV VERBOSE_STARTUP=true

RUN pip3 install celery==5.2.7 unidecode==1.3.6 requests==2.31.0

COPY scripts/*.py /

COPY orthanc.json /etc/orthanc/
