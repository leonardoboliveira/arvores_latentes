FROM python:3.6

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt
RUN apt-get update
RUN apt-get install default-jdk -y
RUN python -m nltk.downloader wordnet

ENV PATH /usr/bin/cort:$PATH