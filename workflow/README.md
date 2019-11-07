## Install Docker 
https://docs.docker.com/install/

## Clone the Optimal github project
git clone https://github.com/MaastrichtU-IDS/OptimAL.git

## Go to the main folder in the optimal and Build Docker 
cd OptimAL \
docker build -t optimal .

## Run docker image
docker run --rm -it -p 8891:8899 optimal
## Open Jupyter notebook and paste your token
http://127.0.0.1:8899/ 

## Change the config this line in optimal-config.yaml 
xml_path: "Your folder containing XML files (drug labels) "

## Enter your API Key in src/BPAnnotator.py
API_KEY ="Replace this with your given BioPortal API Key. If you do not have one you can get one at (https://bioportal.bioontology.org/)"

## Open Termianl in Jupyter and Run workflow
cwl-runner --outdir indi workflow/optimal-pipeline.cwl workflow/optimal-config.yaml 

