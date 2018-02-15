# Argumentative Zoning corpus loading and graph generating
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from __future__ import absolute_import
from __future__ import print_function
import re
import codecs
from bs4 import BeautifulStoneSoup
import json
import difflib
import itertools

all_azs=set([u'OTH', u'BKG', u'BAS', u'CTR', u'AIM', u'OWN', u'TXT']) # argumentative zones
all_ais=set([u'OTH', u'BKG', u'OWN']) # intellectual attribution

rxauthors=re.compile(r"(<REFERENCE>.*?\n)(.*?)(<DATE>)", re.IGNORECASE | re.DOTALL)
rxtitle=re.compile(r"(</DATE>.*?\n)(.*?)(\.|</REFERENCE>)", re.IGNORECASE | re.DOTALL)
rxdate=re.compile(r"(<DATE>)(.*?)(</DATE>)", re.IGNORECASE | re.DOTALL)

rxsingleauthor=re.compile(r"(<SURNAME>)(.*?)(</SURNAME>)", re.IGNORECASE | re.DOTALL)
rxsingleyear=re.compile(r"\d{4}\w{0,1}", re.IGNORECASE | re.DOTALL)
rxwtwoauthors=re.compile(r"(\w+)\sand\s(\w+)", re.IGNORECASE | re.DOTALL)
rxetal=re.compile(r"(\w+)\set\sal", re.IGNORECASE | re.DOTALL)


def loadFileText(filename):
	f=codecs.open(filename,"rb","utf-8", errors="ignore")
	#f=open(filename,"r")

	lines=f.readlines()
	text=u"".join(lines)

##	import unicodedata
##	text = unicodedata.normalize('NFKD', text).decode('UTF-8', 'ignore')

	f.close()
	return text

def writeFileText(text,filename):
	f2=codecs.open(filename,"w","utf-8",errors="ignore")
	f2.write(text)
	f2.close()


def most_common(L):
	"""
		returns the most common element in a list

		from http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
	"""
	groups = itertools.groupby(sorted(L))
	def _auxfun(xxx_todo_changeme):
		(item, iterable) = xxx_todo_changeme
		return len(list(iterable)), -L.index(item)
	return max(groups, key=_auxfun)[0]

def getListOfSentenceObjects(document, headers=False):
	"""
		Returns a list of all the sentences in the document
	"""
	res=[]
	for section in document["sections"]:
		if headers:
			res.append(section["header"])
		for p in section["paragraphs"]:
			for s in p["sentences"]:
				res.append(s)
	return res

def processReference(ref, doc):
	"""
		Process stupid reference format, try to recover title, authors, date
	"""

	lines=ref.__repr__()

	lines=lines.replace("<reference> ","<reference>\n")
	match=rxauthors.search(lines)
	authors=match.group(2).replace("\n","") if match else ""

##	surnames=set([x[1] for x in rxsingleauthor.findall(lines)])
	surnames=[x[1] for x in rxsingleauthor.findall(lines)]

	match=rxdate.search(lines)
	date=match.group(2).replace("\n","") if match else ""

	match=rxtitle.search(lines)
	title=match.group(2).replace("\n","") if match else lines

	if authors=="" and len(surnames) > 0: authors=" ".join(list(surnames))

	newref={"text":lines, "authors":authors, "surnames":surnames, "title":title, "year": date}
	doc["references"].append(newref)

def matchInTextReference(intext,doc):
	"""
		Matches an in-text reference with the bibliography

		TODO: make year optional, add cost for edit distance of year
	"""

	authors=[]
	year=rxsingleyear.search(intext)
	if year: year=year.group(0)

	match=rxwtwoauthors.search(intext)
	if match:
		authors.append(match.group(1))
		authors.append(match.group(2))
	else:
		match=rxetal.search(intext)
		if match:
			authors.append(match.group(1))
		else: # not X and X, not et al - single author
			intext=intext.replace(","," ").replace("."," ")
			bits=intext.split()
			authors.append(bits[0])

	for ref in doc["references"]:
		found=False
		if ref["year"]==year:
			for a in authors:
				if a not in [surname for surname in ref["surnames"]]:
					break
			found=True

		if found:
			return ref
	return None

def processPlainTextAuthor(author):
	"""
		Returns a dictionary with a processed author's name, as
        {"family","given"(,"middlename")}
	"""

##	print author

	bits=author.split()
	res={"family":"","given":"", "text":author}

##	if "<surname>" in author.lower():
##		match=rxsingleauthor.search(author)
##		if match:
##			surname=match.group(2)

	if len(bits) > 1:
		res["family"]=bits[-1]
		res["given"]=bits[0]
		if len(bits) > 2:
			res["middlename"]=" ".join(bits[1:-2])
	elif len(bits) == 1:
		res["family"]=bits[0]
	else:
		pass

	return res

azs=[]
ias=[]


def loadAZannot(filename):
	"""
		Load an AZ-annotated document from the Teufel corpus into a "scidoc" JSON file
	"""

	def loadStructureProcessPara(p, glob):
		glob["p"]+=1
		newPar={"type":"p", "id":glob["p"]}
		newPar["sentences"]=[]

		for s in p.findChildren("s"):
			newSent={"type":"s","text":s.text,"ia":s.get("ia",""),"az":s.get("az",""),"id":glob["s"],"refs":[]}
			newSent["refs"]=[{"text":r.text, "link":0} for r in s.findAll("ref")]
			glob["s"]+=1
			newPar["sentences"].append(newSent)

		return newPar

	def loadStructureProcessDiv(div, doc, glob):
		header=div.find("header")

		newSection={"header":header, "paragraphs":[], "id":glob["sect"]}
		glob["sect"]+=1
		for p in div.findAll("p"):
			newPar=loadStructureProcessPara(p,glob)
			newSection["paragraphs"].append(newPar)

		doc["sections"].append(newSection)

	glob={"sect":0,"p":0,"s":0}


	f=codecs.open(filename,"rb","utf-8", errors="ignore")
	lines=f.readlines()
	text="".join(lines)
	soup=BeautifulStoneSoup(text)

	paper=soup.find("paper")
	title=paper.find("title").text

	newDocument={"title":title}
	newDocument["sections"]=[]
	newDocument["references"]=[]
	newDocument["metadata"]={"fileno":paper.find("fileno").text}

	authors=[]
	meta=soup.find("metadata")
	for a in meta.findChildren("author"):
		authors.append(processPlainTextAuthor(a.text))

	newDocument["authors"]=authors
	newDocument["year"]=meta.find("year").text

	for ref in soup.findAll("reference"):
		processReference(ref, newDocument)

	newSection={"header":"Abstract", "paragraphs":[], "id":glob["sect"]}
	glob["sect"]+=1
	newSection["paragraphs"].append({"type":"p", "sentences":[], "id":glob["p"]})
	glob["p"]+=1

	abstract=soup.find("abstract")
	for s in abstract.findChildren("a-s"):
		newSent={"type":"s","text":s.text,"ia":s["ia"],"az":s["az"],"id":glob["s"], "refs":[]}
		newSection["paragraphs"][-1]["sentences"].append(newSent)
		glob["s"]+=1

	newDocument["sections"].append(newSection)

	for div in soup.findAll("div"):
		loadStructureProcessDiv(div, newDocument, glob)

	sentences=getListOfSentenceObjects(newDocument)
	for s in sentences:
		for ref in s["refs"]:
			match=matchInTextReference(ref["text"],newDocument)
			if match:
##				print ref["text"]," -> ", match["authors"], match["year"]
##				print s.get("az","NO AZ")
##				print s.get("ia","NO IA")
				azs.append(s.get("az","NO AZ"))
				ias.append(s.get("ia","NO IA"))
				match["AZ"]=match.get("AZ",[])
				match["AZ"].append(s.get("az","OTH"))
				match["IA"]=match.get("AZ",[])
				match["IA"].append(s.get("az",""))
			else:
				print("NO MATCH for CITATION in REFERENCES:", ref["text"])
				pass

## "in press", "forthcoming", "submitted", "to appear"
# No functiona por: unicode
##	for ref in newDocument["references"]:
##		k=ref.get("AZ",["NO AZ"])
##		print k, most_common(k)

	return newDocument

def findMatchingReferenceByTitle(ref, docs):
	"""
		Returns the element in the docs collection that best matches the supplied input reference or None if not found
	"""
	threshold=0.9

	best=0.0
	res=None

	for d in docs:
		if d["year"]==ref["year"]: # silly comparison, this doesn't necessarily work
			s=difflib.SequenceMatcher(None,d["title"].lower(),ref["title"].lower())
			r=s.quick_ratio()
			if r > threshold and r > best:
				res=d

	return res

def authorOverlap(authors,surnames):
	"""
		Computes a 0 to 1 score for author overlap
	"""
	match=0

	for s in surnames:
		for a in authors:
			au=a["family"].lower()
			s=s.lower()
			if s in au:
				match+=1
				break

	if len(surnames) > 0:
		return match/float(len(surnames))
	else:
		return 0


def findMatchingReferenceByAuthors(ref, docs):
	"""
		Returns the element in the docs collection that best matches the supplied input reference or None if not found
	"""
	threshold=0.95

	best=0.0
	res=None

	for d in docs:
		if d["year"]==ref["year"]: # silly comparison, this doesn't necessarily work
			if authorOverlap(d["authors"],ref["surnames"]) > .95:
				return d


	return None

##	matches=difflib.get_close_matches(ref["title"],[d["title"] for d in docs],1,.1)
##	if len(matches) > 0:
##		return matches[0]
##	return None

def generateGraph(docs):
	"""
		Generates a dict that is dumpable to a JSON file to be loaded for D3 visualization
	"""
	nodes=[]
	links=[]

	group=1
	for doc in docs:
		group+=.2
		nodes.append({"name":generateFullInfo(doc),"group":round(group)})
		doc["graph_num"]=len(nodes)-1


	for doc in docs:
##		nodes[doc["title"]]={"borders":1}
		for r in doc["references"]:
##			print r["title"]
			if r["authors"] == "":
				print("  NO AUTHORS!",r)
			match=findMatchingReferenceByAuthors(r,docs)
			if match:
				if doc.get("graph_num",0)==0:
					group+=1
 					nodes.append({"name":generateFullInfo(doc),"fullinfo":generateFullInfo(doc),"group":1})
					doc["graph_num"]=len(nodes)-1

				#print "  MATCH! ", match["title"], match["metadata"]["fileno"]
##				nodes.append({"name":r["title"],"group":2})
##				links.append({"source":doc["graph_num"],"target":len(nodes),"value":1})
				if match.get("graph_num",0)==0:
## 					nodes.append({"name":match["title"],"group":2})
 					nodes.append({"name":generateFullInfo(match),"group":2})
					match["graph_num"]=len(nodes)-1
				links.append({"source":doc["graph_num"],"target":match["graph_num"],"type":most_common(r.get("AZ",["TXT"]))})
			else:
##				print "NO MATCH for REFERENCE in CORPUS:", r
				pass

	print("AZs", set(azs))
	print("AIs", set(ias))
	return nodes, links


def inTextReference(refValues):
	"""
		Return an in-text reference
	"""
	authors=refValues.get('authors',[])
	res=""
	if authors == []:
		res+="?"
	elif len(authors) == 1:
		res += '%s ' % authors[0]["family"]
	elif len(authors) == 2:
		res += '%s and %s' % (authors[0]["family"], authors[1]["family"])
	else:
		res += '%s et al.' % authors[0]["family"]

	res+=" (%s)" % refValues["year"]
	return res

def generateFullInfo(doc):
	res=""
	authors=[]
	for s in doc["authors"]:
		authors.append(s)
	res+="<strong>%s</strong>" % inTextReference({"authors":authors, "year":doc["year"]})
##	print "!!", authors, res
	res+="<br/><br/>"
	res+="<em>%s</em>" % doc["title"]
	res+="<br/><br/>FILE: " + doc["metadata"]["fileno"]
##	print res
	return res

def loadAZcorpus(dir):
	import glob
	annots=glob.glob(dir)

	nodes=[]
	links=[]

	docs=[]
	for a in annots:
		doc=loadAZannot(a)
		docs.append(doc)

	return docs


def main():
	docs=loadAZcorpus(r"C:\NLP\PhD\raz\input\*.annot")
	nodes,links=generateGraph(docs)

	data={"nodes":nodes,"links":links}
	writeFileText(json.dumps(data),r"C:\NLP\PhD\raz\graph\d3\output2.json")


if __name__ == '__main__':
    main()
