import pandas as pd

DM = pd.read_csv('XMLProduct.csv')
DB = pd.read_csv('Raw/drugbank vocabulary.csv')
DB = DB[['DrugBank ID', 'UNII']]
DB.columns = ['DB_ID', 'UNII_ID']

#These two datasets should have the same name
newDM = DM.merge(DB, on=["UNII_ID"], how = 'inner')

newDM.to_csv("GSD_DBID.csv")