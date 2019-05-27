# OptimAL

Download and install all libraries within the requirements.txt file before continuing.


# Note: need to construct a requirements.txt file
```
pip install requirements.txt
```

Step 1: 
Download druglabel data from DailyMed: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

Recommended: Test this pipeline on one of the monthly updates. This will contain the newest druglabels and a smaller subset to run everything faster for the first time.

a) Place the .zip file into the "DailyMedExtracter" folder and extract the prescriptiopn folder here. As a result there should be 1,000+ zip files within this folder.
b) Place all of these zip files into the DailyMedExtracter folder.
c) Run the "DailyMed_extracter.sh" in your commandline.

This should obtain the initital csv file "DailyMedXML2CSV.csv" which is primarily made up of the DailyMed xml files and uses the DBID from Drugbank.

Run this from the terminal in order to move the file into the correct directory for further opertations.
```
mv DailyMedExtracter/temp_xml/DailyMedXML2CSV.csv Datasets/XMLProduct.csv
```

Following this use the .csv file to merge the dataset and obtain the DrugBankID using the DBIDMerge.sh:

```
sh DBIDMerge.sh
```

Now we have obtained the DBID from the UNII code. This will allow us to obtain the feature matrix later on.

Now we need to clean up the text, this is important as we want multiple people viewing and collecting the metadata from these labels

```
TODO: Convert manual text cleaning to automatic text cleanup
```

Now that the text is magically cleaned, gather the possible annotations from the bioportal annotator, use the DOID ontology to obtain the associated UMLS ID and merge the two dataframes together to get the final product for crowdsourcing.

```
sh BPannotation.sh
```

