import itertools
import math
from csv import reader
import sys
import argparse

if __name__ =="__main__":

	parser =argparse.ArgumentParser()
	parser.add_argument('-g', required=True, dest='goldindications', help='enter path to file for drug indication gold standard ')
	parser.add_argument('-o', required=True, dest='output', help='enter path to the file for feature matrix ')
	parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
	parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

	args= parser.parse_args()

	goldindfile=args.goldindications
	outfile=args.output
	drugfeatfiles=args.drugfeat
	diseasefeatfiles=args.diseasefeat
	

	indClass=dict()
	drugs = set()
	diseases =set()

	#with open( '../data/input-data/indications_drugcentral.txt' ) as fileDrugIndKnown:
	#with open( '../data/input-data/all_association_umls_drugcentral.txt' ) as fileDrugIndKnown:
	#filename=sys.argv[1]
	
	with open( goldindfile ) as fileDrugIndKnown:
	# skip header
		header=next(fileDrugIndKnown)
		#header =header.replace("\"","").split(",")
		header = header.strip().split("\t")
		for line in fileDrugIndKnown:
			line = line.strip().replace('"','').split("\t")
			#print line
			disease=line[1]

			drug=line[0]
			#cls =int(line[2])
			indClass[(drug,disease)]=1
			drugs.add(line[0])
			diseases.add(line[1])


	drugFeatureSet=[]
	drugFeatures={}
	#for featureFilename in ['../data/features/drugs-smiles.txt','../data/features/drugs-targets.txt']:
	for featureFilename in drugfeatfiles:
		featureFile =file(featureFilename)
		header =featureFile.next()
		#print header
		header =header.strip().split()
		#print header
		featureNames=header[1:]
		#print featureNames
		for name in featureNames:
			drugFeatureSet.append(name)
		for line in featureFile:
			line = line.strip().split()
			if not drugFeatures.has_key(line[0]):
				drugFeatures[line[0]]={}

			for i in range(len(featureNames)):
				var=featureNames[i]
				#print var
				drugFeatures[line[0]][var]=line[1+i]



	cellFeatureSet=[]
	cellFeatures={}
	#for featureFilename in ['../data/features/diseases-meddra.txt']:
	#for featureFilename in ['../data/features/diseases-meddra-hpo.txt','../data/features/diseases-meddra.txt']:
	for featureFilename in diseasefeatfiles:
		#print featureFilename
		featureFile =file(featureFilename)
		header =featureFile.next()
		#print header
		header =header.strip().split('\t')
		#print header
		featureNames=header[1:]
		#print featureNames
		for name in featureNames:
			cellFeatureSet.append(name)
		for line in featureFile:
			line = line.strip().split('\t')
			if not cellFeatures.has_key(line[0]):
				cellFeatures[line[0]]={}

			for i in range(len(featureNames)):
				var=featureNames[i]
				#print var
				cellFeatures[line[0]][var]=line[1+i]

	sep ='\t'
	featureHeader =sep.join(drugFeatureSet)+sep+sep.join(cellFeatureSet)
	trainfile = open(outfile,'w')
	trainfile.write("Drug"+sep+"Disease"+sep+featureHeader+sep+"Class\n")
        commonDrugs= drugs.intersection( drugFeatures.keys())
	commonDiseases= diseases.intersection(cellFeatures.keys())
	for dr in commonDrugs:
 		for di in commonDiseases:
			pair=(dr,di)
			cls=0
			if pair in indClass: cls =1
		
			row = dr+sep+di
			#feauturemat =cellFeatures[id2cellLine[di]]
			if  not drugFeatures.has_key(dr): 
				continue
			#drugFeatures[dr]={}

			if  not cellFeatures.has_key(di):
				continue
			#cellFeatures[di]={}

			feauturemat =drugFeatures[dr]

		
	
			for feature in drugFeatureSet:
				#print feature
				if feauturemat.has_key(feature):
					row+= sep+feauturemat[feature].strip()
				else:
					row+=sep+"0"


			feauturemat =cellFeatures[di]
		
			for feature in cellFeatureSet:
				#print feature
				if feauturemat.has_key(feature):
					row+= sep+feauturemat[feature].strip()
				else:
					row+=sep+"0"

			row += sep+str(cls)
			trainfile.write( row+'\n' )
				


