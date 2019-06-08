FROM ubuntu:18.04
FROM python:3
RUN pip install Pillow & pip install opencv-python & pip install numpy
COPY . /app
CMD python /app/train.py