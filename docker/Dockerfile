FROM python:3.6

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

RUN mkdir /code
RUN mkdir /code/data_handling
RUN mkdir /code/scripts
RUN mkdir /data
RUN mkdir /data/conll
RUN mkdir /data/enconded
RUN mkdir /model
RUN mkdir /model/encoder