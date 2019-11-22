import pandas as pd
import argparse



def DBIDMerge(input_file, output_file, mapping_file):    
    #Input File
    DB = pd.read_csv(mapping_file)
  
    
    DB = DB[['DrugBank ID', 'UNII', 'Common name']]
    DB.rename(columns={'DrugBank ID':'DB_ID', 'UNII':'UNII_ID', 'Common name': 'Drug_name'},inplace=True)
    
    #The total length of the Drugbank drug count and the number of Drugs based off of the active ingredient/Unii ID
    DB.dropna(subset=["UNII_ID"], inplace=True)
    print("The total number of UNIQUE DBID gathered from the Drugbank website is: ",len(DB))
    
    
    
    #File to add DBID to:
    DM = pd.read_csv(input_file)
    DM.dropna(subset=["Text"], inplace=True)
    
    #The total length of the DailyMed label count and the number of unique labels based off of the active ingredient/Unii ID
    print("The total number of labels gathered from the DailyMed website is:       ",len(DM))
    print("The total number of UNIQUE labels gathered from the DailyMed website is:", len(DM.drop_duplicates(['UNII_ID'])))
    
    #Drop the instances from DailyMed if they share the same UNII ID
    #DM.drop_duplicates(['UNII_ID'], inplace=True, keep='first')

    #These two datasets should have the same name
    newDM = DM.merge(DB, on=["UNII_ID"], how = 'inner')
    
    #DM = DM[DM["WordCount"] < 200]
    DM = DM.sort_values(by = "WordCount", ascending = False)
    
    print ("The number of drug labels are reduced from ",len(DM)," to", len(newDM))
    
    
    newDM.to_csv(output_file, index=False)
    
if __name__ =="__main__":
    
    parser =argparse.ArgumentParser()
    parser.add_argument('-i', required=False, default= "../data/output/XMLProduct_cleaned.csv", dest='input', help='enter the code from which type of label you want')
    parser.add_argument('-o', required=False, default="../data/output/unlabeled_withDBID.csv", dest='output', help='output path in order to define where the xmlproduct should be written')
    parser.add_argument('-m', required=False, default='../data/input/drugbank vocabulary.csv', dest='mapping', help=' enter the mapping file for Drugbank (drugbank vocabulary.csv)' )

    args= parser.parse_args()
    input_file = args.input
    output_file = args.output
    mapping_file =args.mapping
    
    DBIDMerge(input_file, output_file, mapping_file)
