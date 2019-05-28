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
	# the protein targets are listed with '*' denoting any character 

	targetdict =dict()
	alltargets =[]
	targetFreq =dict()
	for row in infile:
		row =row.strip().replace('"','').split('\t')
		drugid = row[0]
		target = row[1]
		#print "before",drugid+"\t"+str(targets)
		if targetdict.has_key(drugid):
			targetdict[drugid].append(target)
		else:
			targetdict[drugid] = [target]
		alltargets.append(target)

	#alltargets= [ t for t in targetFreq if targetFreq[t] >1 ]        
	uniqueTargets=sorted(set(alltargets))
	header ="Drug"
	for t in uniqueTargets:
		header+="\tgeneid:"+t
	print header
	
	for drug1 in sorted(targetdict):
		targetList1 = targetdict[drug1]
		featureStr = drug1
		for t in uniqueTargets:
			if t in targetList1 :
				featureStr+="\t1"
			else:
				featureStr+="\t0"
		print featureStr

