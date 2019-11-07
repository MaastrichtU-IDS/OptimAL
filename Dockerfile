FROM dclong/jupyterlab
  
USER root

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    build-essential \
    curl

RUN pip install --upgrade pip && \
  pip3 install -r requirements.txt
