import urllib.request, urllib.error, urllib.parse
import json
import os
import pandas
import argparse
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
                          

    
if __name__ =="__main__":
    
    parser =argparse.ArgumentParser()
    parser.add_argument('-i', required=False, default= "../data/output/unlabeled_withDBID.csv", dest='input', help='enter the code from which type of label you want')
    parser.add_argument('-o', required=False, default="'../data/output/unlabeled_withBPAnnotations.csv'", dest='output', help='output path in order to define where the xmlproduct should be written')
    args = parser.parse_args()

    #Input the data you want to work with in here
    df = pandas.read_csv(args.input)
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

    data = []
    for index, row in df.iterrows():
        
        if index % 100  == 0:
            perc = index/len(df) *100
            print(str(index) + " : "+ str(perc))
        
        #Text input for the ontology search engine
        text_to_annotate = row["Text"]
        db_id = row["DB_ID"]
        drug_name = row['Active_ingredient']
        drug_brand_name = row["Drug_Brand_Name"]
        label_id = row["Label_ID"]

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
                    data.append([label_id, annotation["from"], annotation["to"], annotation["matchType"], annotation["text"], 
                                 text_to_annotate, class_details["@id"], db_id, drug_name, drug_brand_name ])
        except:
            pass


    #Constructs the new dataframe (newdf) from the collected lists_
    columns =['Label_ID','From','To','Type', 'Annotation', 'Context','DO_ID','DB_ID','DrugName', 'DrugBrandName' ]

    newdf = pandas.DataFrame(data, columns= columns)

    #Length of each of the df and the average number of annotations made per label
    numOfAnno = len(newdf)
    numOfContext = len(df)

    print(numOfAnno)
    print(numOfContext)

    print(numOfAnno/numOfContext)   
    newdf.to_csv(args.output, index=False)
                              
                         