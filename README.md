# OptimAL

Download and install all libraries within the requirements.txt file before continuing.


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

Step 7: Use the REMZI Pipeline in order to obtain predictions on the most informative instances

```
python drugindication_ml/src/train_and_test.py -g drugindication_ml/data/input/unified-gold-standard-umls.txt -t data/output/final_unlabeled.csv -dr drugindication_ml/data/features/drugs-fingerprint.txt drugindication_ml/data/features/drugs-targets.txt -di drugindication_ml/data/features/diseases-ndfrt-meddra.txt -m rf -p 2 -s 100 -o data/output/predictions_for_unlabeled.csv
```
#NOTE: Why did we remove one of the feature matrixes again? Also we are not using the selected features matrix. 


Step 8: Make up the dataset using positive and negative examples

Go to the match relations file in order to ignore instances which are already within the goldstandard dataset

```
python Match_RelationsFromLabel+GoldStd.py
python GatherNegativeExamples.py
```

