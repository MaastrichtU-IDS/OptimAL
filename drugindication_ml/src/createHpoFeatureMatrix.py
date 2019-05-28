import sys
import re
from csv import reader


if __name__== '__main__':
	#if sys.argv is None or len(sys.argv) is not 2:
	#	print "Usage : python convertDrug.. in_file "
	#	exit()
	
	drugdict =dict()
	
	
	infile = file(sys.argv[1])
	#infile = file("../data/DREAM10/Drug_info_release.csv")
	header=infile.next()
	# the protein hpos are listed with '*' denoting any character 

	hpodict =dict()
	allhpos =[]
	hpoFreq =dict()
	for row in infile:
		row =row.strip().replace('"','').split('\t')
		#print row
		drugid = row[0]
		hpo = row[1]
		#print "before",drugid+"\t"+str(hpo)
		if hpodict.has_key(drugid):
			hpodict[drugid].append(hpo)
		else:
			hpodict[drugid] = [hpo]
		allhpos.append(hpo)

	#allhpos= [ t for t in hpoFreq if hpoFreq[t] >1 ]        
	uniqueTerms=sorted(set(allhpos))
	header ="Drug"
	for t in uniqueTerms:
		header+="\t"+t
	print header
	
	for drug1 in sorted(hpodict):
		hpoList1 = hpodict[drug1]
		featureStr = drug1
		for t in uniqueTerms:
			if t in hpoList1 :
				featureStr+="\t1"
			else:
				featureStr+="\t0"
		print featureStr

