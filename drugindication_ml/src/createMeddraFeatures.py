import sys
import re
from csv import reader


if __name__== '__main__':
	#if sys.argv is None or len(sys.argv) is not 2:
	#	print "Usage : python convertDrug.. in_file "
	#	exit()
	
	drugdict =dict()
	
	takeName=0
	takeInt=0
	takeTarget=0
	label=""
	
	infile = file(sys.argv[1])
	#infile = file("../data/DREAM10/Drug_info_release.csv")
	header=infile.next()
	# the protein meddras are listed with '*' denoting any character 

	meddradict =dict()
	allmeddras =[]
	meddraFreq =dict()
	for row in infile:
		row =row.replace('"','').replace("http://bio2rdf.org/",'').strip().split('\t')
		drugid = row[0]
		term = row[1]
		
		if meddradict.has_key(drugid):
			meddradict[drugid].append(term)
		else:
			meddradict[drugid] = [term]
		#allmeddras.append(term)
		if meddraFreq.has_key(term): 
			meddraFreq[term]+=1
		else:
			meddraFreq[term]=1

	allmeddras= [ t for t in meddraFreq if meddraFreq[t] >0 ]        
	uniqueMeddras=sorted(set(allmeddras))
	#uniqueMeddras.remove("")
	#uniqueMeddras.remove("http://www.w3.org/2002/07/owl#Thing")
	header ="Disease"
	for t in uniqueMeddras:
		header+="\t"+t
	print header
	
	for drug1 in sorted(meddradict):
		meddraList1 = meddradict[drug1]
		featureStr = drug1
		for t in uniqueMeddras:
			if t in meddraList1 :
				featureStr+="\t1"
			else:
				featureStr+="\t0"
		print featureStr

