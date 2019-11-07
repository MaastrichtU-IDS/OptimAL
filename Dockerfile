FROM jupyter/pyspark-notebook
  
USER root
WORKDIR /jupyter

COPY requirements.txt  ./

RUN apt-get update && apt-get install -y wget ca-certificates build-essential curl cwltool

RUN pip install --upgrade pip && \
  pip install -r requirements.txt
