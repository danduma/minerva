# Rename AZ files to their ACL IDs. Am I still using this?
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import os, glob, shutil

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

def main():
	renameAZFilesToACL("C:\NLP\PhD\RAZ-master\input\\")
	pass

if __name__ == '__main__':
    main()
