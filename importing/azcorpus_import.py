# take the .scixml and .annot files from AZ and CFC corpora, import them to SciDocJson
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import os, glob, shutil
from xmlformats.azscixml import *

#===============================
#       OLD CORPUS RENAMING
#===============================

def loadRenameList(filename):
	"""
	"""
	mappings={}

	f=open(filename,"r")
	for line in f:
		if line[0] != "#":
			parts=line.split()
			if len(parts)==1:
				parts.append(parts[0])
			mappings[parts[0]]=parts[1]
	return mappings

def renameAZFilesToACL(dir):
	"""
	"""
	file_mask=dir+"*.annot"
	AZdocs=[a for a in glob.glob(file_mask)]

	renamelist=loadRenameList(r"C:\NLP\PhD\RAZ-master\mappings.txt")

	for az in AZdocs:
		num=os.path.basename(az).lower().replace(".annot","")
		newnum=renamelist[num]
		print "copy ",az,dir+newnum+".xml"
		print "copy ",az.replace(".annot",".txt"),dir+newnum+".txt"
		shutil.copyfile()


def testTypesContained(container):
    """
        Will return a set with all the types contained in a container (list or dict)
    """
    def recurseTypes(cont2, typeset):
        typeset.add(type(element))
        if isinstance(cont2,list):
            elements=cont2
        elif isinstance(element, dict):
            elements=element.values()
        else:
            return

        for element in elements:
            recurseTypes(element,typeset)

    types=set()
    recurseTypes(container,types)
    return types

def convertAnnotToSciDoc(input_mask,output_dir):
    """
        Given a mask "c:\bla\*.annot" it will load each file and save it in
        output_dir as a SciDoc .JSON
    """
    output_dir=ensureTrailingBackslash(output_dir)
    for filename in glob.glob(input_mask)[2:]:
        print "Converting",filename
        doc=loadAZSciXML(filename)
        fn=os.path.basename(filename)
        doc.saveToFile(output_dir+os.path.splitext(fn)[0]+".json")



def main():
##	renameAZFilesToACL("C:\NLP\PhD\RAZ-master\input\\")
	pass

if __name__ == '__main__':
    main()
