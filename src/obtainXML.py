from zipfile import ZipFile
import os


#Method for selecting specific files from a zip file (in this case xml files since we only look at files ending in a '.xml')
def extract(filename, path, to_path):    
	filename =   path + filename
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
    parser.add_argument('-p', required=False, default= "../dailymed/prescription4/", dest='path', help='enter the directory where druglabels reside (should be .zip)')
    parser.add_argument('-o', required=False, default="../data/output/unlabeled_withDBID.csv", dest='output', help='output folder in order to define where the xml files should be saved')

    args= parser.parse_args()
    path = args.path
    output_folder = args.output

    #Method for going thorugh all individual zip files within the choosen directory and extracting information from it
    count = 0
    #path = "../dailymed/prescription4/"
    #Goes through all filenames in the current folder
    for filename in os.listdir(path):
        #Only interacts with zip files (files ending with ".zip")
        if filename.endswith('.zip'):
            count = count + 1
            print ('processing ', filename) 
            #Needs a unique name for file we will now extract
            strcount = "temp_xml"+ str(count)
            extract(filename, path, output_folder)
