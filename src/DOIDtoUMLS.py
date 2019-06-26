import pandas as pd

df = pd.read_csv('../data/output/unlabeled_withBPAnnotations.csv')

print("The number of annotations from the bioportal annotator is:",  len(df))

# Download and process UMLS to DOID mappings

#This is the updated DOID2UMLS mapping with 5974 unique DOID identifiers
domap_df = pd.read_csv("../data/input/DOID2UMLS.csv")
print(len(domap_df))
print(len(domap_df.drop_duplicates("doi_id")))

def trimName(url):
    return  url.replace("http://purl.obolibrary.org/obo/","")

df.DO_ID= df.DO_ID.apply(trimName)


df.drop_duplicates(['DO_ID', 'DB_ID'], inplace=True)

domap_df = domap_df[['doi_id', 'umls_id']]   
domap_df.rename(columns={'doi_id':'DO_ID','umls_id':'disease'}, inplace=True)

print(len(df))
print(len(domap_df))

#Merge the DOID mapping to the BPAnnotator
df2 = df.merge(domap_df, on=["DO_ID"], how='inner')


df2.drop_duplicates(["DO_ID", "DB_ID"], inplace=True)
df2.rename(columns={'DB_ID':'Drug','disease':'Disease'},inplace=True)

print("The final size of the unlabeled data is:"+str(len(df2)))

df2.to_csv("../data/output/final_unlabeled.csv", index=False)