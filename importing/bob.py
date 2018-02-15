# processing of bob corpus. Deprecated.
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import codecs
import re
import os

rxfileno=re.compile(r"(<FILENO>)(.*?)(</FILENO>)", re.IGNORECASE | re.DOTALL)
rxdocno=re.compile(r"(<DOCNO>)(.*?)(</DOCNO>)", re.IGNORECASE | re.DOTALL)

rxreftype=re.compile(r"(<REF\s)(.*?)(>)", re.IGNORECASE | re.DOTALL)

uniques={}

def exploreBobCorpus(filename):
	"""
	"""

	f=codecs.open(filename,"rb","utf-8", errors="ignore")

	count=0

	for line in f:
		match=rxfileno.search(line)
		if not match:
			match=rxdocno.search(line)

		if match:
			fileno=match.group(2).replace("\n","")
			print(fileno)
			count+=1


	f.close()
	print("Total files:", count)


def listvalues(filename,regex):
	"""
		Goes through the file matching a regex to a line and printing the value if match found
	"""
	f=codecs.open(filename,"rb","utf-8", errors="ignore")

	for line in f:
		match=regex.search(line)

		if match:
			fileno=match.group(2).replace("\n","")
			print(fileno)


	f.close()

def listuniquevalues(filename,regex):
	"""
		Goes through the file matching a regex to a line and printing the value if match found
	"""
	f=codecs.open(filename,"rb","utf-8", errors="ignore")

	for line in f:
		match=regex.search(line)

		if match:
			fileno=match.group(2).replace("\n","")
			uniques[fileno]=uniques.get(fileno,0)+1

	f.close()

	for u in uniques:
		print(u,uniques[u])

def splitMassiveBobIntoFiles(filename, subdir):
	"""
		What is says on the tin.
	"""

	f=codecs.open(filename,"rb","utf-8", errors="ignore")
	dir=os.path.dirname(filename)
	dir=os.path.dirname(dir)+"\\"+subdir
	if not os.path.exists(dir):
		os.makedirs(dir)

	count=0

	newdoc=""

	for line in f:
		if count >= 10000:
			break

		if line.strip().lower()=="</doc>":
			match=rxfileno.search(newdoc)
			if not match:
				match=rxdocno.search(newdoc)

			if match:
				fileno=match.group(2).replace("\n","")
			else:
				fileno="NO_MATCH_"+str(count)

			print("Saving file #",count, " - ", fileno)
			try:
				f2=codecs.open(dir+"\\"+fileno+".xml","wb","utf-8", errors="ignore")
				f2.write(newdoc)
				f2.close()
			except:
				fileno="NAME_ERROR_"+str(count)
				f2=codecs.open(dir+"\\"+fileno+".xml","wb","utf-8", errors="ignore")
				f2.write(newdoc)
				f2.close()

			newdoc=""
 			count+=1

		else:
			newdoc+=line

		#print line
	f.close()

def fixBobXML(infile,outfile):
	"""
		One-time fix of many annoying problems with bob XML
	"""
	max_reasonable_header=70
	glob_s_id=0
	glob_footnote_id=0

	fulltext=loadFileText(infile)

	# FIX METADATA: ADD FROM EXTERNAL SOURCE

	# FIX PARAGRAPHS:
	# join paragraphs that are broken sentences

	# first sentence of paragraph starts with lowercase, remove <p> paragraph boundaries
	changes=1
	while changes > 0:
		fulltext, changes=re.subn(r"(</[pP]>\s*?<[pP]>)(\s*?<[sS].{0,20}?\">)([a-z])",r"\2\3", fulltext)

	# (</[sS]>\s*</[pP]>\s*?<[pP]>)(.\s*?<[sS].{0,20}?\">)([a-z])
	# join small non-paragraphs (formulas, tables, etc) of only one short non-sentence
	changes=1
	while changes > 0:
		fulltext, changes=re.subn(r"(</s>)\s*?</p>\s*?<p>\s*?<s\sid.{0,10}>(.{0,30})</s>\s*?</p>",r"\2\1", fulltext, re.IGNORECASE|re.DOTALL)

	# FIX SENTENCES
	# sentence too short?
	# several sentences in a single <s>?

	# HEADER problems
	# match a header that is too long
	header_matches=[m for m in re.finditer(r"<header\sid.{3,12}>",fulltext,re.DOTALL|re.IGNORECASE)]
	for header_match in reversed(header_matches):
		closing_header=re.search(r"</header>",fulltext[header_match.end():],re.IGNORECASE)
		if not closing_header: continue
		header=fulltext[header_match.end():header_match.end()+closing_header.start()]
		if len(header) <= max_reasonable_header:
			continue

		match1=re.match(r"\s*\d\.\d(\.\d)*.*",header)
		if match1:
			# 'tis a header with more stuff in it
			match1=re.search(r"\s*\d\.\d(\.\d)*",header)
			start=match1.end()
			match2=re.search(r"\.",header[start+1:])
			if match2:
				actual_header=cleanxml(header[start:match2.end()+1])
				print("actual header", actual_header)
				para_text=header[match2.end()+1:]
				glob_s_id+=1
				para_text="<p><s id='s-ins-"+str(glob_s_id)+"'>"+para_text+"</s></p>"
				print("para text",para_text)
				fulltext=fulltext[:header_match.start(1)]+actual_header+"</header>"+para_text+fulltext[header_match.end(0):]
		elif re.match(r"\d\.?\s*.*",header):
			# 'tis a footnote, dammit! Fix it!
			glob_footnote_id+=1
			glob_s_id+=1
			footnote_text="<p><s id='s-ins-"+str(glob_s_id)+"'><footnote id='"+str(glob_footnote_id)+"'>"+header+"</footnote></s></p>"
			fulltext=fulltext[:header_match.start(0)]+footnote_text+fulltext[header_match.end()+closing_header.end():]

	writeFileText(fulltext,outfile)


def main():
##	exploreBobCorpus(r"C:\NLP\PhD\bob\bob\documents\documents.xml")
	splitMassiveBobIntoFiles(r"C:\NLP\PhD\bob\bob\documents\documents.xml", "files2")
##	listuniquevalues(r"C:\NLP\PhD\bob\bob\documents\documents.xml",rxreftype)

	pass

if __name__ == '__main__':
    main()
