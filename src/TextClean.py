import pandas as pd
import string
import re


def cleanText(text): 
    return text.replace('|','\n')

def removeCharacters(s):
    b = "®ÃÂ¢â€ž¢®â‰¥"
    for char in b:
        a = s.replace(char,"")
    return a
def removeParagraphs(s):
    s = s.replace('\\n',' ')
    return s
def removeSlashes(s):
    s = s.replace("\'",'')
    return s
def removeDuplicatedWhitespace(s):
    " ".join(s.split())
    return s
def removePatterns(s):
    return re.sub(r'\\t',  '',  s)
def removePatterns2(s):
    return re.sub(r'\\xa0',  '',  s)
def removePatterns3(s):
    return re.sub(r'\\x',  '',  s)
def removePatterns4(s):
    return re.sub(r'â‰¥',  '',  s)
def removeCommaSpace(s):
    return re.sub('                     ,',  '',  s)
def removeNones(s):
    return re.sub('None,', '', s)
def removeNones2(s):
    return re.sub('None],', ']', s)
def trailingwhitespace(s):
    return s.strip()




df = pd.read_csv("../data/output/unlabeled_withDBID.csv")
print("The number of rows which " +str(len(df)))

df.rename(columns={"Context":"Text"}, inplace=True)

#Need to drop NAN values to avoid errors
df.dropna(subset = ["Text"], inplace=True)


df.Text = df.Text.apply(removeCommaSpace)
df.Text = df.Text.apply(removeNones)
df.Text = df.Text.apply(removeNones2)
df.Text = df.Text.apply(removePatterns)
df.Text = df.Text.apply(removePatterns2)
df.Text = df.Text.apply(removePatterns3)
df.Text = df.Text.apply(removePatterns4)
df.Text = df.Text.apply(removeCharacters)
df.Text = df.Text.apply(removeParagraphs)
df.Text = df.Text.apply(removeSlashes)
df.Text = df.Text.apply(removeDuplicatedWhitespace)
df.Text = df.Text.apply(trailingwhitespace)

counter = []
for index, row in df.iterrows():
    test = row['Text']
    #Counts the number of entries at row x and adds it to the counter list
    counter.append(len(test.split()))

#The word count list is now appended to the context file
df['WordCount'] = counter

#Sorts instances by string length
df = df.sort_values(by = "WordCount", ascending = False)

df = df[df.WordCount>4]
print(len(df))
df.head()

df.to_csv("../data/output/unlabeled_withDBID.csv", index=False)