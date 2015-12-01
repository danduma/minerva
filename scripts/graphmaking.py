#-------------------------------------------------------------------------------
# Name:        graphmaking.py
# Purpose:      Graph-making functions, moved here from main scixml.py
#
# Author:      Daniel Duma
#
# Created:     14/12/2013
# Copyright:   (c) Daniel Duma 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from scixml import *
from reference_formatting import *

# ------------------------------------------------------------------------------
#   Graph generating helper functions
# ------------------------------------------------------------------------------


def makeDocFullInfo(doc):
	"""
		Returns extensive HTML-formatted info about a document for popup display

		doc - loaded SciXML
	"""
	res=""
	authors=[]
	for s in doc["authors"]:
		authors.append(s)
	res+=u"<strong>%s</strong>" % formatCitation({"authors":authors, "year":doc["year"]})
##	print "!!", authors, res
	res+=u"<br/><br/>"
	try:
		res+=u"<em>%s</em>" % doc["title"]
	except:
		res+=u"<em>%s</em>" % unicode(doc["title"],errors="replace")
	if doc.has_key("filename"):
		res+=u"<br/><br/>FILE: " + doc["filename"]
##	print res
	return res


def generateRTTreeGraph(docname, outputfilename, index, working_dir, maxdepth=5, include_not_in_collection=True):
	"""
		Generates a dict that is dumpable to a JSON file to be loaded for D3 visualization
		of RTT radial tree

		doc - name of SciXML file
	"""

	allhashes={}

	def addDocToGraph(filename, graph, depth):
		"""
		"""

		doc=loadSciXML(filename,True, True, False)

		node={}
		allhashes[normalizeTitle(doc["title"])]=node
		node["name"]=makeAPAReference(doc)
		node["id"]=len(allhashes)
		node["filename"]=filename
		node["fulldescription"]=makeDocFullInfo(doc)
		node["children"]=[]

		if depth < maxdepth:
			for ref in doc["references"]:
				print "processing reference", ref["surnames"], ref["year"], ref["title"]
				title=normalizeTitle(ref["title"])
				if title in allhashes:
					node["children"].append(
					{"name":"POINTER: "+makeAPAReference(ref),
					"fulldescription":makeDocFullInfo(ref),
					"pointer":allhashes[title]["id"]})
				else:
					match=matchReferenceInIndexHash(ref, index)
					if match:
						addDocToGraph(working_dir+match["filename"], node["children"], depth+1)
					else:
						if include_not_in_collection:
							node["children"].append(
							{"name":makeAPAReference(ref),
							"fulldescription":makeDocFullInfo(ref)})
		else:
			node["cropped"]=True

		graph.append(node)

		print "Added node for ", doc["authors"], doc["year"], doc["title"]
		return graph


	tree=addDocToGraph(working_dir+docname, [], 0)
	tree=tree[0]

	if outputfilename:
		writeFileText(json.dumps(tree),outputfilename)
	return tree

# ------------------------------------------------------------------------------
#   Graph generating functions
# ------------------------------------------------------------------------------

def generateGraph(docs):
	"""
		Generates a dict that is dumpable to a JSON file to be loaded for D3 visualization
	"""
	nodes=[]
	links=[]

	group=1
	for doc in docs:
		group+=.2
		nodes.append({"name":makeDocFullInfo(doc),"group":round(group)})
		doc["graph_num"]=len(nodes)-1


	for doc in docs:
##		nodes[doc["title"]]={"borders":1}
		for r in doc["references"]:
##			print r["title"]
			if r["authors"] == "":
				print "  NO AUTHORS!",r
			match=findMatchingReferenceByAuthors(r,docs)
			if match:
				if doc.get("graph_num",0)==0:
					group+=1
 					nodes.append({"name":makeDocFullInfo(doc),"fullinfo":makeDocFullInfo(doc),"group":1})
					doc["graph_num"]=len(nodes)-1

				#print "  MATCH! ", match["title"], match["metadata"]["fileno"]
##				nodes.append({"name":r["title"],"group":2})
##				links.append({"source":doc["graph_num"],"target":len(nodes),"value":1})
				if match.get("graph_num",0)==0:
## 					nodes.append({"name":match["title"],"group":2})
 					nodes.append({"name":makeDocFullInfo(match),"group":2})
					match["graph_num"]=len(nodes)-1
				links.append({"source":doc["graph_num"],"target":match["graph_num"],"type":most_common(r.get("AZ",["TXT"]))})
			else:
##				print "NO MATCH for REFERENCE in CORPUS:", r
				pass

	print "AZs", set(azs)
	print "AIs", set(ias)
	return nodes, links




def testMakeGraph():
	"""
	"""
	hashed=loadIndex(r"C:\NLP\PhD\bob\bob\hashed.pic")
	count=1
	start=0
	for doc in DOC_LIST[start:2]:
		generateRTTreeGraph(doc,r"C:\Users\MasterMan\Dropbox\PhD\code\graphs\out"+str(count+start)+".json",hashed,
		r"C:\NLP\PhD\bob\bob\files\\",maxdepth=10, include_not_in_collection=False)
		count+=1


def main():
    pass

if __name__ == '__main__':
    main()
