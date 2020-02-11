## Install Docker 
https://docs.docker.com/install/

## Clone the Optimal github project
git clone https://github.com/MaastrichtU-IDS/OptimAL.git

## Go to the main folder in the optimal and Build Docker 
cd OptimAL \
docker build -t optimal .

## Run docker image
docker run --rm -it --name optimal -p 8891:8888 -v $(pwd):/jupyter optimal

## Change the config parameters in workflow/optimal-config.yaml 
xml_path: "Your folder containing XML files (drug labels) "
api_key ="Replace this with your given BioPortal API Key." If you do not have one you can get one at (https://bioportal.bioontology.org/)

## Open Terminal from Jupyter and Run workflow
docker exec -it optimal cwltool --outdir output workflow/optimal-pipeline.cwl workflow/optimal-config.yaml 

