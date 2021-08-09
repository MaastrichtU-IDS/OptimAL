# OptimAL 
## Relation Extraction from DailyMed Structured Product Labels by Optimally Combining Crowd, Experts and Machines
The effectiveness of machine learning models to provide accurate and consistent
results in drug discovery and clinical decision support is strongly dependent
on the quality of the data used. However, substantive amounts of open data
that drive drug discovery suffer from a number of issues including inconsistent
representation, inaccurate reporting, and incomplete context. For example,
databases of FDA-approved drug indications used in computational drug
repositioning studies do not distinguish between treatments that simply o er
symptomatic relief from those that target the underlying pathology. Moreover,
drug indication sources often lack proper provenance and have little overlap.
Consequently, new predictions can be of poor quality as they o er little in the
way of new insights. Hence, work remains to be done to establish higher quality
databases of drug indications that are suitable for use in drug discovery and
repositioning studies. Here, we report on the combination of weak supervision
(programmatic labeling, crowdsourcing) and deep learning methods for relation
extraction from DailyMed text to create a higher quality drug-disease relation
dataset. The generated drug-disease relation data shows a high overlap with
DrugCentral, a manually curated dataset. Using this dataset, we constructed
a machine learning model to classify relations between drugs and diseases from
text into four categories; treatment, symptomatic relief, contradiction, and effect,
exhibiting an improvement of 15.5 % with Bi-LSTM (F1 score of 71.8% )
over the best performing discrete method. Access to high quality data is crucial
for building accurate and reliable drug repurposing prediction models. Our
work suggests how the combination of crowds, experts, and machine learning
methods can go hand-in-hand to improve datasets and predictive models.

## Download and install 
use the requirements.txt file to install requirements


```
pip install requirements.txt
```

Step 1: Download the data 
Download the druglabel data from DailyMed: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
For this take the full prescription data from the full releases. At the moment this is made up of 4 zip files. Take these and extract these zip files into a prescription folder.

Note: Test this pipeline on one of the monthly updates. This will contain the newest druglabels and a smaller subset to run everything faster for the first time.


Step 2: Extract the information
Each .zip file represents an SPL label. In order to extract the information we want, we must first extract the xml files from the .zip files.

```
cd OptimAL\src

python obtainXML.py

python DailymedXMLExtracter.py
```
This will extract the xml files from all of the zip files and then DailymedExtracter.py will go through the content of each of these xmls if they contain an "Indications and Usage" section.
As a result, we should recieve a file called "XMLProduct.csv".

Step 3: Gather the DrugBank ID using the UNII code
Using the active ingredient and Unii code extracted from the XML files, use this to gather the DB_ID

```
python DBIDmerge.py
```
From this we will recieve the "GSD_DBID.csv" file. 


Step 4: Clean up the text
Make the gathered context text even neater by using this code to make the text look nicer:

```
python StringLength.py
python TextClean.py
```



We clean up the text now so that we make sure we can get annotations from the BioPortal annotation API as well as make it look neater for the future microtasks.

Step 5: Gather the BioPortal annotations

```
python BPAnnotator.py
```



Step 6: 

Gather the UMLS ID using the DOID Mappings. Without these UMLS ID we cannot run the Drug Repurposing Pipeline (DRP)

```
python DOID2UMLSMapping.py
```
