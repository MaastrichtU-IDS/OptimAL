import urllib.request, urllib.error, urllib.parse
import json
import os
import pandas
from pprint import pprint

REST_URL = "http://data.bioontology.org"
API_KEY = "a28f1d5b-0cc4-454a-8baf-1b2285cfa549"

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

def print_annotations(annotations, get_class=True):
    #Result is the row
    for result in annotations:
        #returns a martix called annotatedClass which holds all of the ids, links, ontologies, etc
        class_details = result["annotatedClass"]
        if get_class:
            try:
                class_details = get_json(result["annotatedClass"]["links"]["self"])
            except urllib.error.HTTPError:
                print(f"Error retrieving {result['annotatedClass']['@id']}")
                continue
        print("Class details")
        print("\tid: " + class_details["@id"])
        print("\tprefLabel: " + class_details["prefLabel"])
        print("\tontology: " + class_details["links"]["ontology"])

        print("Annotation details")
        for annotation in result["annotations"]:
            print("\tfrom: " + str(annotation["from"]))
            print("\tto: " + str(annotation["to"]))
            print("\tmatch type: " + annotation["matchType"])

        if result["hierarchy"]:
            print("\n\tHierarchy annotations")
            for annotation in result["hierarchy"]:
                try:
                    class_details = get_json(annotation["annotatedClass"]["links"]["self"])
                except urllib.error.HTTPError:
                    print(f"Error retrieving {annotation['annotatedClass']['@id']}")
                    continue
                pref_label = class_details["prefLabel"] or "no label"
                print("\t\tClass details")
                print("\t\t\tid: " + class_details["@id"])
                print("\t\t\tprefLabel: " + class_details["prefLabel"])
                print("\t\t\tontology: " + class_details["links"]["ontology"])
                print("\t\t\tdistance from originally annotated class: " + str(annotation["distance"]))

        print("\n\n")


                        
#Input the data you want to work with in here
df = pandas.read_csv('GSD_DBID.csv')
print(len(df))
#Now to make this run for each context available!!!!
id = []
From = []
To = []
matchType = []
annotation2 = []
ontology = []
context = []
dbid = []
drugname = []

#Adds additional parameters here for the bioportal search engine
additional_parameters = "&ontologies=DOID&require_exact_match=true"

for index, row in df.iterrows():
    
    if index  % 200 == 0:
        perc = index/len(df) *100
        print(str(index) + " : "+ str(perc))
    
    #Text input for the ontology search engine
    text_to_annotate = row["Text"]
    db_id = row["DB_ID"]
    drug_name = row["Active_ingredient"]
    
    

    try:
        # Annotate using the provided text
        annotations = get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(text_to_annotate) + additional_parameters)

        for result in annotations:
            class_details = result["annotatedClass"]

            for annotation in result["annotations"]:
                From.append(annotation["from"])
                To.append(annotation["to"])
                matchType.append(annotation["matchType"])
                annotation2.append(annotation["text"])   
                context.append(text_to_annotate)
                ontology.append("\tontology: " + class_details["links"]["ontology"])
                id.append(class_details["@id"])
                dbid.append(db_id)
                drugname.append(drug_name)
    except:
        pass

#Constructs the new dataframe (newdf) from the collected lists
newdf = pandas.DataFrame({'Id':id})
newdf['Drug Name'] = drugname
newdf['DB_ID'] = dbid
newdf['Context'] = context
newdf['Ontology'] = ontology
newdf['From'] = From
newdf['To'] = To
newdf['Type'] = matchType
newdf['Text'] = annotation2    


#Length of each of the df and the average number of annotations made per label
numOfAnno = len(newdf)
numOfContext = len(df)

print(numOfAnno)
print(numOfContext)

print(numOfAnno/numOfContext)                              
                          
#Now use the gathered annotations and use the DOID mapping to find the disease UMLS
df = newdf

# Download and process UMLS to DOID mappings
# We use the propagated mappings here
url = 'https://raw.githubusercontent.com/dhimmel/disease-ontology/72614ade9f1cc5a5317b8f6836e1e464b31d5587/data/xrefs-prop-slim.tsv'
domap_df = pandas.read_table(url)
domap_df = domap_df.query('resource == "UMLS"')
domap_df['diseaseId'] = domap_df['resource_id']
domap_df = domap_df[['doid_code', 'doid_name', 'diseaseId']]

                          
x = []

for index, row in df.iterrows():
    y = row["Id"].split("_")
    x.append(y[1])

#df["DOID"] = x
df.insert(3, 'DOID', x)

x = []

for index, row in domap_df.iterrows():
    y = row["doid_code"].split(":")
    x.append(y[1])

domap_df["DOID"] = x


domap_df = domap_df[['doid_name', 'diseaseId', 'DOID']]   

#Merge the DOID mapping to the BPAnnotator
df2 = df.merge(domap_df, on=["DOID"], how='inner')

df2.to_csv("BPAUMLS.csv")
                          
                          
#Merge the BPAnnotations with the XML file
df3 = pandas.read_csv("GSD_DBID.csv")
df4 = df2.merge(df3, on = ["DB_ID"], how = 'inner')
df5 = df4[['Label_ID', 'Drug_Brand_Name', 'Active_ingredient', 'Text_y', 'UNII_ID', 'DB_ID', 'Ontology', 'From', 'To', 'Text_x', 'diseaseId']]
df5.columns = ['Label ID', 'Drug Brand Name', 'Active Ingredient', 'Context', 'UNII ID', 'DBID', 'Ontology', 'From', 'To', 'Text', 'UMLS ID']
                          
                          
df5.to_csv("FinalCSV.csv")