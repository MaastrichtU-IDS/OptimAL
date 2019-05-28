
# Sources
-------
1- **Drugbank**

RDF Dataset Obtained from: http://download.bio2rdf.org/release/4/drugbank/drugbank.nq.gz

Original complete Dataset Obtained from : https://www.drugbank.ca/releases/latest

2- **Kegg**

RDF Dataset Obtained from: http://download.bio2rdf.org/release/4/kegg/kegg-drug.nq.gz
                       http://download.bio2rdf.org/release/4/kegg/kegg-genes.nq.gz
                       
Original complete Dataset Obtained from : ftp://ftp.genome.jp/pub/kegg/

3- **SIDER**

RDF Dataset Obtained from: http://download.bio2rdf.org/release/4/sider/sider-se.nq.gz

Original complete Dataset Obtained from:  http://sideeffects.embl.de/media/download/

4- **NDF-RT**

Original complete Dataset Obtained from: http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT_XML.zip

5- **MEDDRA**

RDF Dataset Obtained from: http://purl.bioontology.org/ontology/MEDDRA

6- **Drug Indication from NDR-FT and DrugCentral**

RDF Data generated: data/input/unified-gold-standard-umls.nq.gz

------------------------------------
DRUG TARGETS -from DRUGBANK and KEGG 
------------------------------------
```bash
PREFIX iv: <http://bio2rdf.org/irefindex_vocabulary:>
PREFIX dv: <http://bio2rdf.org/drugbank_vocabulary:>
PREFIX hv: <http://bio2rdf.org/hgnc_vocabulary:>
PREFIX kv: <http://bio2rdf.org/kegg_vocabulary:>
SELECT distinct ?drugid ?geneid
WHERE
{
?drug <http://bio2rdf.org/openpredict_vocabulary:indication> ?i .
{
  ?d a kv:Drug .
  ?d kv:target ?l .
  ?l kv:link ?t .
  BIND (URI( REPLACE(str(?t),"HSA","hsa")) AS ?target) .
  ?target a kv:Gene .
  ?target kv:x-ncbigene ?ncbi .
  #?target kv:x-uniprot ?ncbi .
  ?d kv:x-drugbank ?drug .
}
UNION
{
  ?drug a dv:Drug .
  ?drug dv:target ?target .
  ?target dv:x-hgnc ?hgnc .
  ?hgnc hv:x-ncbigene ?ncbi .
  #?hgnc hv:uniprot ?ncbi .
}
BIND ( STRAFTER(str(?ncbi),"http://bio2rdf.org/ncbigene:") AS ?geneid)
BIND( STRAFTER(str(?drug), "http://bio2rdf.org/drugbank:") AS ?drugid)
}
```

------------------------------
DRUG SMILES -- from DRUGBANK
------------------------------

```bash
PREFIX dv: <http://bio2rdf.org/drugbank_vocabulary:>

SELECT distinct ?drugid ?smiles
{
 ?d <http://bio2rdf.org/openpredict_vocabulary:indication> ?i .
 ?d dv:calculated-properties ?cp .
 ?cp a dv:SMILES .
 ?cp dv:value ?smiles .
 BIND( STRAFTER(str(?d), "http://bio2rdf.org/drugbank:") AS ?drugid)
}
```

------------------------------
DRUG SIDE EFFECTS - from SIDER
------------------------------

```bash
PREFIX dv: <http://bio2rdf.org/drugbank_vocabulary:>
PREFIX sider: <http://bio2rdf.org/sider_vocabulary:>

SELECT distinct ?drugid ?se
{
?dia <http://bio2rdf.org/sider_vocabulary:drug> ?stitch_flat .
?dia <http://bio2rdf.org/sider_vocabulary:effect> ?se .
{
?stitch_flat <http://bio2rdf.org/sider_vocabulary:x-pubchem.compound> ?pc .
}
UNION{
?stitch_flat <http://bio2rdf.org/sider_vocabulary:stitch-stereo> ?stitch_stereo .
?stitch_stereo <http://bio2rdf.org/sider_vocabulary:x-pubchem.compound> ?pc .
}
?d dv:x-pubchemcompound ?pc  .
BIND(STRAFTER( str(?d), "http://bio2rdf.org/drugbank:") AS ?drugid)
}
```
------------------------------------------
DISEASE DESCRIPTIONS - from MEDDRA, NDF-RT 
-----------------------------------------
```bash
PREFIX sider: <http://bio2rdf.org/sider_vocabulary:>
SELECT distinct ?umlsid ?parent_id
WHERE {
?drug <http://bio2rdf.org/openpredict_vocabulary:indication> ?i .
?a <http://bioportal.bioontology.org/ontologies/umls/cui> ?i .
?a rdfs:subClassOf* ?parent .
FILTER (regex(str(?parent),"^http://bio2rdf.org/ndfrt:")
 || regex(str(?parent),"^http://bio2rdf.org/meddra:") )
BIND (STRAFTER(str(?i),"http://bio2rdf.org/umls:") AS ?umlsid)
BIND (STRAFTER(str(?parent),"http://bio2rdf.org/") AS ?parent_id)
}

```
------------------------------------------


```bash
export SPARQL_ENDPOINT=http://localhost:13065/sparql
#query drug target info and downlad in the input folder
curl -H "Accept: text/tab-separated-values" --data-urlencode query@drugbank-drug-target.sparql $SPARQL_ENDPOINT > drugbank-drug-target.tab

#query drug smiles info and downlad in the input folder
curl -H "Accept: text/tab-separated-values" --data-urlencode query@drugbank-drug-smiles.sparql $SPARQL_ENDPOINT > drugbank-drug-smiles.tab

curl -H "Accept: text/tab-separated-values" --data-urlencode query@sider-meddra-terms.sparql $SPARQL_ENDPOINT > sider-meddra-terms.tab

```


# How to Run
## Requirement
```bash
python 3.6
scikit-learn
pandas
numpy 
```



## Create binary feature matrix for drug and disease

```bash
python createTargetFeatureMatrix.py ../data/input/drugbank-drug-target.tab > ../data/features/drugs-targets.txt
#create feature matrix for drug fingerprint 
python createFingerprintFeatures.py ../data/input/drugbank-drug-smiles.tab > ../data/features/drugs-fingerprint.txt
#disease meddra features
python createMeddraFeatures.py ../data/input/sider-meddra-terms.tab > ../data/features/diseases-meddra.txt
```

## Cross-Validation

 
do 10-fold cross-validation, separate gold standard into train and test by removing 10% of drugs and theirs association
train on training set and test on test set and report the performance of each fold
```bash
 python cv_test.py -g ../data/input/unified-gold-standard-umls.txt -dr ../data/features/drugs-targets.txt ../data/features/drugs-fingerprint.txt ../data/features/drugs-sider-se.txt -di ../data/features/diseases-ndfrt-meddra.txt -o ../data/output/completeset_unified_validation.txt -disjoint 0 -p 2 -m rf
 ```
Command Line

```
usage: cv_test.py -g GOLDINDICATIONS -m MODELTYPE -disjoint DISJOINT -o
                  OUTPUT -p PROPORTION -dr DRUGFEAT [DRUGFEAT ...] -di
                  DISEASEFEAT [DISEASEFEAT ...]

-g		enter path to the file for drug indication gold standard 
-dr	 	paths to the files for drug features
-di		enter paths to the files for disease features
-m		classification model name ( logistic | knn | tree | rf | gbc )
-disjoint	enter disjoint [0,1,2] (0: Pair-wise, 1 : Drug-wise 2: Disease-wise Disjoint)
-o 		path to output file for model
-p		number of proportion for negative samples

 
 ```


## Predict new probablitities of unknown relations

train with whole gold standard set

```bash
python train_and_test.py -g ../data/input/unified-gold-standard-umls.txt -t ../data/predictions/PREDICT-repositioned-drug-mapped.csv -dr ../data/features/drugs-targets.txt ../data/features/drugs-fingerprint.txt ../data/features/drugs-sider-se.txt -di ../data/features/diseases-ndfrt-meddra.txt -s 405 -o ../data/predictions/rf_p2_n405.txt -m rf -p 2

usage: train_and_test.py -g GOLDINDICATIONS -t TEST -m MODELTYPE -o OUTPUT -p
                         PROPORTION -s SEED -dr DRUGFEAT [DRUGFEAT ...] -di
                         DISEASEFEAT [DISEASEFEAT ...]

-g		enter path to the file for drug indication gold standard
-t		enter path to the file you want to get predictions for
-dr	 	paths to the files for drug features
-s		enter seed number 
-di		enter paths to the files for disease features
-m		classification model name ( logistic | knn | tree | rf | gbc )
-o 		path to output file for model
-p		number of proportion for negative samples

 ```

