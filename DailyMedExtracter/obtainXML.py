from zipfile import ZipFile
import os

def extract(filename, num):
    
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(filename, 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for fileName in listOfFileNames:
            # Check filename endswith csv
            if fileName.endswith('.xml'):
                # Extract a single file from zip
                zipObj.extract(fileName, 'temp_xml')
    
    return



count = 0

#Goes through all filenames in the current folder
for filename in os.listdir('.'):
    #Only interacts with zip files (files ending with ".zip")
    if filename.endswith('.zip'):
        count = count + 1

        #Needs a unique name for file we will now extract
        strcount = "temp_xml"+ str(count)
        extract(filename, strcount)
