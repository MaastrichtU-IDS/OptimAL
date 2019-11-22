from zipfile import ZipFile
import os
import argparse
import zipfile


#Method for selecting specific files from a zip file (in this case xml files since we only look at files ending in a '.xml')
def extract(filename, path, to_path):    
	filename =   os.path.join(path, filename)
	# Create a ZipFile Object and load sample.zip in it
	with ZipFile(filename, 'r') as zipObj:
		# Get a list of all archived file names from the zip
		listOfFileNames = zipObj.namelist()
		# Iterate over the file names
		print ('Unziping')
		for fileName in listOfFileNames:
			print(fileName)
			if fileName.endswith('.xml'):
				# Extract a single file from zip
				zipObj.extract(fileName, to_path)


if __name__ =="__main__":
    
    parser =argparse.ArgumentParser()
    parser.add_argument('-p', required=False, dest='path', help='enter the directory where druglabels reside (should be .zip)')
    parser.add_argument('-o', required=False, dest='output', help='output folder in order to define where the xml files should be saved')

    args= parser.parse_args()
    path = args.path
    output_folder = args.output
    directory_to_extract_to = 'temp'
    xml_directory =os.path.join(directory_to_extract_to,'prescription')

    #Method for going thorugh all individual zip files within the choosen directory and extracting information from it
    count = 0
    #path = "../dailymed/prescription4/"
    #Goes through all filenames in the current folder
    
    #for zip_file in os.listdir(path):
    #    with zipfile.ZipFile( os.path.join(path, zip_file), 'r') as zip_ref:
    #        zip_ref.extractall(directory_to_extract_to)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for filename in os.listdir(xml_directory):
        #Only interacts with zip files (files ending with ".zip")
        if filename.endswith('.zip'):
            count = count + 1
            print ('processing ', filename) 
            #Needs a unique name for file we will now extract
            extract(filename, xml_directory, output_folder)
    print ("count",count)