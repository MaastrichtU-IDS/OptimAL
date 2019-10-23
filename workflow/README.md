## Install Docker 
https://docs.docker.com/install/

## Pull a docker image from DockeHub for python (eg. amalic/jupyterlab)
docker pull amalic/jupyterlab:latest

## Run docker image
docker run --rm -it -p 8888:8888 amalic/jupyterlab

## Open jupyter and type this on terminal
apt-get update && apt-get install -y     wget     ca-certificates     build-essential     curl  cwltool

## Clone the Optimal github project
git clone https://github.com/MaastrichtU-IDS/OptimAL.git

## Go to the main folder in the optimal 
pip install -r requirements.txt

## Change this line in optimal-config.yaml 
abs_path: "/notebooks/code/OptimAL/"

## Enter your API Key in BPAnnotator.py
API_KEY ="Replace this with your given BioPortal API Key. If you do not have one you can get one at (https://bioportal.bioontology.org/)"

## Run workflow
cwl-runner --outdir indi workflow/optimal-pipeline.cwl workflow/optimal-config.yaml 

