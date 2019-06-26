import pandas as pd
import argparse


def DBIDMerge(output = "../data/output/unlabeled_withDBID.csv", input = "../data/output/XMLProduct.csv"):    
    #Input File
    DB = pd.read_csv('../data/input/drugbank vocabulary.csv')
    
    DB = DB[['DrugBank ID', 'UNII', 'Common name']]
    DB.rename(columns={'DrugBank ID':'DB_ID', 'UNII':'UNII_ID', 'Common name': 'Drug_name'},inplace=True)
    
    #The total length of the Drugbank drug count and the number of Drugs based off of the active ingredient/Unii ID
    print("The total number of DBID gathered from the Drugbank website is:        ",len(DB))
    print("The total number of UNIQUE DBID gathered from the Drugbank website is: ",len(DB.drop_duplicates(['UNII_ID'])))
    
    DB.dropna(subset=["UNII_ID"], inplace=True)
    
    #File to add DBID to:
    DM = pd.read_csv('../data/output/XMLProduct.csv')
    DM.dropna(subset=["Text"], inplace=True)
    
    #The total length of the DailyMed label count and the number of unique labels based off of the active ingredient/Unii ID
    print("The total number of labels gathered from the DailyMed website is:       ",len(DM))
    print("The total number of UNIQUE labels gathered from the DailyMed website is:", len(DM.drop_duplicates(['UNII_ID'])))
    
    counter = []
    for index, row in DM.iterrows():
        test = row['Text']
        #Counts the number of entries at row x and adds it to the counter list
        counter.append(len(test.split()))

    #The word count list is now appended to the context file
    DM['WordCount'] = counter
    DM = DM.sort_values(by = "WordCount", ascending = False)
    
    #Drop the instances from DailyMed if they share the same UNII ID
    DM.drop_duplicates(['UNII_ID'], inplace=True, keep='first')

    #These two datasets should have the same name
    newDM = DM.merge(DB, on=["UNII_ID"], how = 'inner')
    
    print ("The number of drug labels are reduced from ",len(DM)," to", len(newDM))
    
    
    newDM.to_csv(output, index=False)
    
if __name__ =="__main__":
    
    #parser =argparse.ArgumentParser()
    #parser.add_argument('-i', required=True, dest='inputfile', help='enter the code from which type of label you want')
    #parser.add_argument('-o', required=True, dest='output', help='output path in order to define where the xmlproduct should be written')
    
    #args= parser.parse_args()
    #inputfile = args.inputfile
    #output = args.output
    
    DBIDMerge()