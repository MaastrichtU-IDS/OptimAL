import pandas as pd
import string
import re
import os
import pprint as pp
import argparse
from cleantext import clean


def cleanText(text): 
    s= clean(text, fix_unicode=True, to_ascii=True,  lower=False, no_line_breaks=True, lang="en" ) 
    s= re.sub(r'\|+', '|', s)
    s= s.replace('|', '\r\n')
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', required=True, dest='input',help="enter input file path")
    parser.add_argument('-out', required=True, dest='output', help="enter ouput file path")
    args = parser.parse_args()
    input_file_path = args.input
    output_file_path = args.output

    df = pd.read_csv(input_file_path)
    print("The number of rows initially: " +str(len(df)))

    #df.rename(columns={"full_text":"Text"}, inplace=True)


    df.Text = df.Text.apply(cleanText)


    #Need to drop NAN values to avoid errors
    #df.dropna(subset = ["Text"], inplace=True)
    print("The number of rows after dropping empty text: " +str(len(df)))
    counter = []
    for index, row in df.iterrows():
        test = row['Text']
        #Counts the number of entries at row x and adds it to the counter list
        counter.append(len(test.split()))

    #df.Text.str.split('\n', expand=True).stack()
    #The word count list is now appended to the context file
    df['WordCount'] = counter

    #Sorts instances by string length
    df = df.sort_values(by = "WordCount", ascending = False)

    df.to_csv(output_file_path, index=False, encoding='utf-8')