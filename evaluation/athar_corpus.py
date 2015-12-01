# Parser for Awais Athar's Citation Context Corpus
# see http://www.cl.cam.ac.uk/~aa496/citation-context-corpus/
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
from BeautifulSoup import BeautifulStoneSoup
from minerva.util.general_utils import loadFileText
from minerva.scidoc.reference_formatting import formatReference
from minerva.util.nlp_functions import tokenizeText
from string import punctuation

import re, os, json

# These are hand-crafted regexes. A better approach would be to generate them
# automatically, but this is faster for 20 documents and a quick test
paper_citation_regex={
    "A92-1018": r"Cutting.{1,15}199[12]",
    "C98-2122": r"Lins?.{1,3}1998",
    "D07-1031": r"Johnson.{1,10}2007",
    "J90-1003": r"Church.{2,15}1990",
    "J93-1007": r"Smadja.{1,14}199[23]",
    "J96-2004": r"Carletta.{1,15}1996",
    "N03-1003": r"Barzilay.{1,11}200[23]",
    "N04-1035": r"Galley.{1,12}2004",
    "N06-1020": r"McClosky.{1,15}2006",
    "P02-1053": r"Turney.{1,17}2002",
    "P04-1015": r"Collins.{1,15}2004",
    "P04-1035": r"Pang.{1,15}2004",
    "P04-1041": r"Cahill.{1,20}2004",
    "P05-1045": r"Finkel.{1,15}2005",
    "P07-1033": r"Daum.{2,5}III.{1,3}2007",
    "P90-1034": r"Hind[il]e.{1,5}1990",
    "W02-1011": r"Pang.{1,25}2002",
    "W04-1013": r"Lin.{1,10}2004",
    "W05-0909": r"Banerjee and Lavie.{1,5}2005",
    "W06-1615": r"Blitzer.{1,15}2006",
}

class AtharCorpusReader(object):
    """
    """
    def __init__(self):
        """
        """

    def readFile(self, filename):
        """
            Args:
                filename: full path to file to read
        """
        text=loadFileText(filename)
        return self.read(text, filename)


    def read(self, xml, filename):
        """
            Load a document from the Athar corpus

            Args:
                xml: full xml string
        """
##        # this solves a "bug" in BeautifulStoneSoup with "sec" tags
##        BeautifulStoneSoup.NESTABLE_TAGS["sec"]=[]

        soup=BeautifulStoneSoup(xml)

        paper_data_node=soup.find("div",{"class":"dstPaperData"})
        paper_data={
            "id":paper_data_node.text,
            "title":"",
            "authors":"",
        }
        title=paper_data_node.find("div",{"class":"dstPaperTitle"})
        if title:
            paper_data["title"]=title.text

        authors=paper_data_node.find("div",{"class":"dstPaperAuthors"})
        if authors:
            author_chunks=title.text.split(";")
            for author in author_chunks:
                chunks=author.split(",")
                author_dict={"given":chunks[1],"family":chunks[0]}
            paper_data["authors"]=author_dict

##        print(paper_data)

        all_contexts=[]

        documents=soup.findAll("table", {"class":"srcPaper"})
        for document in documents:
            all_lines=[]
            doc_info=document.findChild("td", {"class":"line srcData"})
            match=re.match(r"(\w+\-\w+)\r?\n(.*)\r?\n(.*)",doc_info.get("title"), flags=re.IGNORECASE)
            if not match:
                doc_from_id=doc_info.get("title")
            else:
                doc_from_id=match.group(1)

##            print("Document: ",doc_info.get("title"))
            lines=document.findChildren("td", {"class":["line x", "line oc", "line o", "line n", "line pc", "line p", "line nc"]})
            for index, line in enumerate(lines):
                sentiment_line=line.get("class").replace("line ","")
                if "x" in sentiment_line:
                    sentiment=None
                else:
                    sentiment=sentiment_line

                line=line.get("title")
                line=re.sub(r"\d+:\d+","",line).strip()
                all_lines.append({"sentiment":sentiment,"line":line})


            for index, line in enumerate(all_lines):
                if line["sentiment"] and "c" in line["sentiment"]:
                    old_line=line["line"]
                    line["line"]=re.sub(paper_citation_regex[paper_data["id"]],"__CIT__",line["line"])

                    if len(old_line) == len(line["line"]):
##                        print(old_line,"\n",paper_data["id"],"\n",paper_citation_regex[paper_data["id"]])
##                        raise ValueError("Couldn't substitute citation!")
                        # no citation found, no reason to do anything
                        continue
                    c_from=max(0,index-4)
                    c_to=min(len(all_lines)-1,index+4)
                    context={"doc_to":paper_data["id"],"doc_from":doc_from_id,"lines":all_lines[c_from:c_to]}
                    all_contexts.append(context)

        return all_contexts

def removeURLs(text):
    """
        Replaces all occurences of a URL with an empty string, returns string.
    """
    return re.sub(r"(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?","",text,flags=re.IGNORECASE)

def removeCitations(text):
    """
        Removes ACL-style citation tokens from text
    """
    old_text=""
    while len(old_text) != len(text):
        old_text=unicode(text)
        text=re.sub(r"(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:, *(?:19|20)[0-9][0-9][a-g]?(?:, p.? [0-9]+)?| \((?:19|20)[0-9][0-9][a-g]?(?:, p.? [0-9]+)?\))", "", text)
    return text

def extractWordContextWindow(context, left, right):
    """
        Returns a window-of-words context: list of tokens
    """
    context_text="".join([line["line"] for line in context["lines"]])
    # remove URLS in text (normally footnotes and conversion erros)
    context_text=removeURLs(context_text)
    context_text=removeCitations(context_text)
    tokens=tokenizeText(context_text)
    tokens=[token for token in tokens if token not in punctuation]
    for index,token in enumerate(tokens):
        if token=="__CIT__":
            res=[]
            res.extend(tokens[index-left:index])
            res.extend(tokens[index+1:index+right+1])
            return res
    return None

def extractContextAnnotated(context):
    """
        Returns a context as annotated: list of tokens
    """
    tokens=[]
    for line in context["lines"]:
        sent=line["sentiment"]
        if sent and ("p" in sent or "n" in sent or "o" in sent or "c" in sent):
            clean_line=removeURLs(line["line"]).replace("__CIT__","")
            clean_line=removeCitations(clean_line)
            tokens.extend(tokenizeText(clean_line))
    tokens=[token for token in tokens if token not in punctuation]
    return tokens

def processAtharCorpus(infiles, outfile):
    """
        Loads the Athar .html corpus into a JSON file of contexts

        Args:
            infiles: file mask of HTML files to load
            outfile: name of .json file to write to
    """
    import glob
    reader=AtharCorpusReader()
    all_contexts=[]
    for f in glob.glob(infiles):
        print("Loading ",f)
        contexts=reader.readFile(f)
        all_contexts.extend(contexts)

    json.dump(all_contexts,file(outfile,"w"))

def loadProcessedContexts(infile):
    """
    """
    return json.load(file(infile,"r"))


def main():
##    processAtharCorpus(r"G:\NLP\PhD\citation_context\*.html", r"G:\NLP\PhD\citation_context\all_contexts.json",)
    contexts=loadProcessedContexts(r"G:\NLP\PhD\citation_context\all_contexts.json")

    for context in contexts[:10]:
        print(extractContextAnnotated(context))
        print("\n")
        print(extractWordContextWindow(context, 50, 50))
        print("\n")

    pass

if __name__ == '__main__':
    main()
