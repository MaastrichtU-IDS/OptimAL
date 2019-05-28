from csv import reader
import sys


def getFingerprint(a):
	from cinfony import pybel
	#print a
	#print b
	mol1 = pybel.readstring("smi", a)
	
	return mol1.calcfp('maccs').bits

def chemicalSimilarityOpenBabel(a,b):
	from cinfony import pybel
	#print a
	#print b
	mol1 = pybel.readstring("smi", a)
	mol2 = pybel.readstring("smi", b)
	
	return mol1.calcfp('maccs') | mol2.calcfp('maccs')


def chemicalSimilarityRDKIT(a,b):
	from rdkit import Chem
	from rdkit import DataStructs
	from rddkit.Chem.Fingerprints import FingerprintMols

	ms1 = Chem.MolFromSmiles(a)
	ms2 = Chem.MolFromSmiles(b)
	fp1=FingerprintMols.FingerprintMol(ms1)
	fp2=FingerprintMols.FingerprintMol(ms2)
	return DataStructs.FingerprintSimilarity(fp1,fp2)



if __name__== '__main__':
	#if sys.argv is None or len(sys.argv) is not 2:
	#	print "Usage : python calculateChemStruct.. in_file "
	#	exit()
	#infile = file("../data/DREAM10/Drug_info_release.csv")
	infile = file(sys.argv[1])
	ddi = list()
	smilesdict =dict()
	takeCode=0
	takeName=0
	label=""
	header=infile.next()
	for row in infile:
		row=row.strip().replace('"','').split()
		#row = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', row)
		#if len(row) < 6: continue
		#print row
		drugid = row[0]
		#print row[6]
		#smiles = row[6].split(";")[0]
		smiles = row[1].split(";")[0]
		if smiles == "": continue 
		smilesdict[drugid] =getFingerprint(smiles)
		#print drugid+"\t"+smiles

	#print len(smilesdict)
	#exit()

	count=0
	maccsLength =166
	header="Drug\t"
	for i in range(maccsLength):
		header+="\tFingerprint"+str(i)

	print header
	
	for drug1 in smilesdict:
			fingerprint1=smilesdict[drug1]
			#print fingerprint1
			#print fingerprint2
			feature=""
			for i in range(maccsLength):
				if i in fingerprint1 :
					feature+="\t1"
				else:
					feature+="\t0"	
			print  str(drug1)+feature
