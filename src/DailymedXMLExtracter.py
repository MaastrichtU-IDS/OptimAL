import os
import xmltodict
import pprint
import json
import xml.etree.ElementTree as ET
from lxml import etree
from datetime import date
import pandas as pd

#Methods for extracting data from SPL labels


#http://www.accessdata.fda.gov/spl/stylesheet/spl-common.xsl
namespaces={"v3":"urn:hl7-org:v3",}

def normalize_date(date_string):
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    return date(year,month,day).strftime("%b %d, %Y")

class DrugLabel(object):
    """represents a Drug Label in the SPL format.
    takes one argument, spl_label, which can be either an url or a file path"""

    def __init__(self, spl_label):
        self.label_data = spl_label
        self.xml = etree.parse(spl_label)

    def actives(self): #UNII code
        """returns a list of active compounds"""
        #here converting to a set removes duplicates
        return sorted(list(set(active.text for active in self.xml.xpath("//v3:ingredientSubstance/v3:activeMoiety/v3:activeMoiety/v3:name",namespaces=namespaces))))
    actives.label = "active cmpds"


    def unii(self): #UNII code
        """returns the drug's NDC number"""
        #this xpath query is NOT from the SPL xsl file
        return sorted(list(set(self.xml.xpath("//v3:ingredientSubstance/v3:activeMoiety/v3:activeMoiety/v3:code/@code",namespaces=namespaces))))
    unii.label = "unii"

    def iau(self):
        """returns the indications and usage section"""
        #here converting to a set removes duplicates
        return sorted(list(set(active.text for active in self.xml.xpath("//v3:structuredBody/",namespaces=namespaces))))
    actives.label = "active cmpds"
    

    def start_date(self):
        """returns start marketing date as a strftime formatted python date object"""
        date_string = self.xml.xpath("//v3:subjectOf/v3:marketingAct/v3:effectiveTime/v3:low/@value",namespaces=namespaces)[0]
        return normalize_date(date_string)
    start_date.label = "marketing start date"

    # don't bother - it's None for all labels
    # end_date.label = "marketing end date"
    # def end_date(self):
    # 	"""returns end marketing date as a strftime formatted python date object or the string "None if not defined
    # 	refers to the expiration date of the last lot released to the market
    # 	(from http://spl-work-group.wikispaces.com/file/view/creating_otc_sp_documentsl.pdf)"""
    # 	try:
    # 		date_string = self.xml.xpath("//v3:subjectOf/v3:marketingAct/v3:effectiveTime/v3:high/@value",namespaces=namespaces)[0]
    # 		return normalize_date(datestring)
    # 	except:
    # 		return "None"

    def marketing_category(self):
        """returns the marketing category"""
        return self.xml.xpath("//v3:subjectOf/v3:approval/v3:code/@displayName",namespaces=namespaces)[0]
    marketing_category.label = "marketing category"

    def revision_date(self):
        """returns label revision date"""
        date_string = self.xml.xpath("/v3:document/v3:effectiveTime/@value",namespaces=namespaces)[0]
        return normalize_date(date_string)
    revision_date.label = "revision date"

    def label_type(self):
        """returns the drug label type, typically 'HUMAN OTC DRUG LABEL' or 'HUMAN PRESCRIPTION DRUG LABEL' """
        return self.xml.xpath("//v3:code/@displayName",namespaces=namespaces)[0]
    label_type.label = "label type"


    def ndc(self):
        """returns the drug's NDC number"""
        #this xpath query is NOT from the SPL xsl file
        return self.xml.xpath("//v3:manufacturedProduct/v3:manufacturedProduct/v3:code/@code",namespaces=namespaces)[0]
    ndc.label = "ndc"

    def name(self):
        """returns the drug's name"""
        return self.xml.xpath("//v3:manufacturedProduct/v3:manufacturedProduct/v3:name",namespaces=namespaces)[0].text.replace("\t","").replace("\n","")
    name.label = "name"

    def distributor(self):
        """returns the drug's distributor"""
        return self.xml.xpath("//v3:author/v3:assignedEntity/v3:representedOrganization/v3:name",namespaces=namespaces)[0].text
    distributor.label = "distributor"

    def dosage_form(self):
        """returns the drug's dosage form"""
        return self.xml.xpath("//v3:manufacturedProduct/v3:manufacturedProduct/v3:formCode/@displayName",namespaces=namespaces)[0]
    dosage_form.label = "dosage form"

    #just a helper function for the next two functions, so no label
    def _get_word_list(self, word):
        """returns a list of etree instances of all occurances of 'word','Word' or 'WORD' """
        word = str(word)
        #this query also NOT from the SPL xsl file
        query = "//*[text()[contains(.,'%s') or contains(.,'%s') or contains(.,'%s')]]" %(word.lower(),word.upper(),word.capitalize())
        return self.xml.xpath(query,namespaces=namespaces)

    def test_word(self, word):
        if self._get_word_list(word): return 1
        else: return 0
    #TODO test_word.label = "%s?" %self.test_word.word

    def get_word_section(self, word):
        #finds the first ancestor section and returns section/code/@displayName
        word_section_list = []
        for word in self._get_word_list(word):
            word_section_list.extend(word.xpath("ancestor::v3:section[1]/v3:code/@displayName",namespaces=namespaces))
        if not word_section_list:
            return "n/a"
        return list(set(word_section_list))

    def get_text(self):
        """returns the drug's label"""
        text = self.xml.xpath('//v3:section/v3:code[@code="34067-9"]/..//v3:paragraph',namespaces=namespaces)
        length = len(text)
        newText = []
        for row in range(length):
            newText.append(text[row].text)
        return newText
    
    def get_fullText(self):
        """returns the drug's label"""
        text = self.xml.xpath('//v3:section/v3:code[@code="34067-9"]/..//v3:text//*',namespaces=namespaces)
        length = len(text)
        newText = []
        for row in range(length):
            newText.append(text[row].text)
        return newText

    def get_word_time(self, word):
        """
        returns the LATEST effectiveTime/@date for all of the instances of "word" mentioned
        """
        word_time_list = []
        for word in self._get_word_list(word):
            word_time_list += word.xpath("ancestor::v3:section/v3:effectiveTime/@value",namespaces=namespaces)
        if not word_time_list:
            return "n/a"
        try:
            return max(list(set(normalize_date(date) for date in word_time_list)))
        except:
            return max(list(normalize_date(date) for date in word_time_list))

    def build_url(self):
        """helper function that builds and returns the accessdata.fda.gov URL given the XML file name/directory"""
        #maybe won't work on windows because slash direction?
        uuid = self.label_data.split("/")[-1].split(".")[0]
        return "http://www.accessdata.fda.gov/spl/data/%s/%s.xml" %(uuid,uuid)

    def indications(self, xml):
        indications = []
        drugN = self.name()
        activeCompound = self.actives()
        uniiCode = self.unii()
        
        try:
            activeC = '|'.join(activeCompound)
        except:
            activeC = activeCompound        
        try:
            uniiC = '|'.join(uniiCode)
        except:
            uniiC = uniiCode        
        
        label = self.get_text()
        try:
            text = '|'.join(label)
        except:
            text = label
        
        full_Label = self.get_fullText()
        try:
            context = '|'.join(full_Label)
        except:
            context = full_Label
        
        data = [xml, drugN, activeC, uniiC, text, context]
        inidcations = indications.append(data)  
        return indications

#Code for going through all xml files and extracting all of the knowledge into one pandas dataframe and into a csv file within the folder

count = 0

for filename in os.listdir('.'):
    if filename.endswith('.xml'):
        
        count = count +1
        try:
            DL = DrugLabel(filename)
        except:
            pass
        try:
            indications = DL.indications(filename)

            if count == 1:
                ind = pd.DataFrame(indications, columns=['Label_ID','Drug_Brand_Name', 'Active_ingredient', 'UNII_ID', 'Formatted_Text','Text'])
            else:
                newind = pd.DataFrame(indications, columns=['Label_ID','Drug_Brand_Name', 'Active_ingredient', 'UNII_ID', 'Formatted_Text','Text'])
                ind = ind.append(newind)
        except:
            pass


            
ind.to_csv("XMLProduct.csv")    
