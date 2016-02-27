# Parser for Awais Athar's Citation Context Corpus
# see http://www.cl.cam.ac.uk/~aa496/citation-context-corpus/
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import re, os, json, sys

from string import punctuation
from copy import deepcopy

from BeautifulSoup import BeautifulStoneSoup
from minerva.proc.general_utils import loadFileText
from minerva.scidoc.reference_formatting import formatReference
from minerva.scidoc.citation_utils import removeACLCitations, removeURLs
from minerva.proc.nlp_functions import tokenizeText
from minerva.scidoc.scidoc import SciDoc
from minerva.scidoc.citation_utils import CITATION_FORM
import minerva.db.corpora as cp
from minerva.evaluation.query_generation import QueryGenerator

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

CIT_MARKER="__CIT__"

query_methods={
    "window":{"parameters":[40,50]}
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

##    def loadCitation(self, ref, sentence_id, newDocument, section_id):
##        """
##            Extract all info from <ref> tag, return dictionary
##        """
##        newCit=newDocument.addCitation(sent_id=sentence_id)
##
##        if ref.has_key("citation_id"):
##            res["original_id"]=ref["citation_id"]
##
##        newCit["original_text"]=ref.__repr__()
##        newCit["ref_id"]=cit
##        newCit["parent_section"]=section_id
##
##        if not newDocument.reference_by_id.has_key(cit):
##            # something strange is going on: no reference with that key
##            print ("No reference found with id: %s" % cit)
####            continue
##        else:
##            newDocument.reference_by_id[cit]["citations"].append(newCit["id"])
##
##        res.append(newCit)

    def wrapInSciDoc(self, contexts, doc_from_id, doc_to_id):
        """
            Returns a SciDoc ready to be passed to the standard context_extract
            functions, where each context is a paragraph

            Args:
                contexts: list of context dicts
                doc_from_id: corpus_id of this SciDoc
                doc_to_id: corpus_id of target document (citation)
            Returns:
                SciDoc
        """
        newDocument=SciDoc()
        metadata=cp.Corpus.getMetadataByField("metadata.corpus_id",doc_from_id)
        if metadata:
            newDocument.loadExistingMetadata(metadata)
            assert newDocument.metadata["guid"] != ""
        else:
            newDocument.metadata["guid"]=doc_from_id
            assert newDocument.metadata["guid"] != ""

        newDocument.metadata["corpus_id"]=doc_from_id

        newSection_id=newDocument.addSection("root", "", 0)

        metadata=cp.Corpus.getMetadataByField("metadata.corpus_id",doc_to_id)
        if not metadata:
            raise ValueError("Target document %s is not in corpus!" % doc_to_id)
            return

        ref=newDocument.addExistingReference(metadata)

        ref["corpus_id"]=doc_to_id

        for context in contexts:
            newPar_id=newDocument.addParagraph(newSection_id)
            for line in context["lines"]:
                newSent_id=newDocument.addSentence(newPar_id)
                text=line["text"]
                citations=[]
                if re.search(CIT_MARKER,text):
                    newCit=newDocument.addCitation(newSent_id, ref["id"])
                    text=re.sub(CIT_MARKER, CITATION_FORM % newCit["id"], text)
                    citations.append(newCit["id"])

                sent=newDocument.element_by_id[newSent_id]
                sent["sentiment"]=line["sentiment"]
                sent["text"]=text
                if len(citations) > 0:
                    sent["citations"]=citations

        return newDocument

    def loadContextLines(self,line, all_lines, index, paper_data):
        """
            Given a line with an annotated citation in the document, returns a
            fully extracted context for that citation

            Args:
                line: dict{sentiment,text}
            returns:
                list of lines to add, with text preprocessed to remove citations
        """
        old_line=line["text"]
        line["text"]=re.sub(paper_citation_regex[paper_data["id"]],CIT_MARKER,line["text"])

        if len(old_line) == len(line["text"]):
##            print(old_line,"\n",paper_data["id"],"\n",paper_citation_regex[paper_data["id"]])
##            raise ValueError("Couldn't substitute citation!")
            # no citation found, no reason to do anything
            return None
        c_from=max(0,index-4)
        c_to=min(len(all_lines)-1,index+4)

        lines_to_add=deepcopy(all_lines[c_from:c_to])
        for line in lines_to_add:
            line["text"]=removeURLs(removeACLCitations(line["text"]))

        return lines_to_add

    def loadDocumentNode(self, document_node, paper_data, index):
        """
            Processes a <document> node, which represents the whole contents of
            a citing paper

            Args:
                document_node: the node
            Returns:
                (doc, contexts): (SciDoc, list of contexts of this document)
        """
        all_lines=[]
        doc_info=document_node.findChild("td", {"class":"line srcData"})
        match=re.match(r"(\w+\-\w+)\r?\n(.*)\r?\n(.*)",doc_info.get("title"), flags=re.IGNORECASE)

        if not match:
            # Error in the Athar corpus: source document ID missing
            doc_from_id=u"temp"+unicode(index)
        else:
            doc_from_id=match.group(1).lower()

        doc_to_id=paper_data["id"].lower()

        lines=document_node.findChildren("td", {"class":["line x", "line oc", "line o", "line n", "line pc", "line p", "line nc"]})
        for index, line in enumerate(lines):
            sentiment_line=line.get("class").replace("line ","")
            if "x" in sentiment_line:
                sentiment=None
            else:
                sentiment=sentiment_line

            line=line.get("title")
            line=re.sub(r"\d+:\d+","",line).strip()
            all_lines.append({"sentiment":sentiment,"text":line})

        doc_contexts=[]
        for index, line in enumerate(all_lines):
            if line["sentiment"] and "c" in line["sentiment"]:
                lines_to_add=self.loadContextLines(line, all_lines, index, paper_data)
                if not lines_to_add:
                    continue
                context={"doc_to":doc_to_id,"doc_from":doc_from_id,"lines":lines_to_add}
                doc_contexts.append(context)

        doc=self.wrapInSciDoc(doc_contexts, doc_from_id, doc_to_id)
        return doc, doc_contexts

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
        all_docs=[]
        document_nodes=soup.findAll("table", {"class":"srcPaper"})
        for index,document_node in enumerate(document_nodes):
            try:
                doc, contexts=self.loadDocumentNode(document_node, paper_data, index)
                all_docs.append(doc)
                all_contexts.extend(contexts)
            except ValueError:
                print("Error:", sys.exc_info()[1])
                break
        return all_docs,all_contexts


class AtharQueryGenerator(QueryGenerator):
    """
    """
    def __init__(self, filename, reassign_guids=False):
        """
            Args:
                filename: converted contexts file
        """
        self.docs=json.load(file(filename,"r"))
        for doc_id in self.docs:
            assert doc_id != ""
            # convert the loaded dicts to a SciDoc instance
            self.docs[doc_id]=SciDoc(self.docs[doc_id])
            assert len(self.docs[doc_id].reference_by_id) > 0
            if reassign_guids:
                cp.Corpus.matchAllReferences(self.docs[doc_id])

    def loadSciDoc(self, guid):
        """
        """
        return self.docs.get(guid,None)

    def getResolvableCitations(self, guid, doc):
        """
            Returns the citations_data of resolvable citations and outlinks
        """
        res=[]
        sents_with_multicitations=[]

        for cit in doc["citations"]:
            res.append({"cit":cit,
                        "match_guid":doc.reference_by_id[cit["ref_id"]]["guid"],
                        })

        for sent in sents_with_multicitations:
            # deal with many citations within characters of each other: make them know they are a cluster
            doc.countMultiCitations(sent)

        citations_data={"resolvable":res,"outlinks":doc["references"][0]}
        return citations_data

    def loadDocAndResolvableCitations(self, guid):
        """
            Overrides

            Args:
                guid: self-explanatory
            Returns:
                (doc, doctext, precomputed_file)
        """
        doc=self.loadSciDoc(guid) # load the SciDoc JSON from the corpus
        if not doc:
            raise ValueError("ERROR: Couldn't load SciDoc: %s" % guid)
            return None

        doctext=doc.getFullDocumentText() #  store a plain text representation

        # load the citations in the document that are resolvable, or generate if necessary
        citations_data=self.getResolvableCitations(guid, doc)

        return doc,doctext,citations_data


def getOutlinkContextAtharWindowOfWords(context, left, right):
    """
        Returns a window-of-words context: list of tokens
    """
    context_text="".join([line["line"] for line in context["lines"]])
    # remove URLS in text (normally footnotes and conversion erros)
    context_text=removeURLs(context_text)
    context_text=removeACLCitations(context_text)
    tokens=tokenizeText(context_text)
    tokens=[token for token in tokens if token not in punctuation]
    for index,token in enumerate(tokens):
        if token==CIT_MARKER:
            res=[]
            res.extend(tokens[index-left:index])
            res.extend(tokens[index+1:index+right+1])
            return res
    return None

def getOutlinkContextAtharAnnotated(context):
    """
        Returns a context as annotated: list of tokens
    """
    tokens=[]
    for line in context["lines"]:
        sent=line["sentiment"]
        if sent and ("p" in sent or "n" in sent or "o" in sent or "c" in sent):
            clean_line=removeURLs(line["line"]).replace(CIT_MARKER,"")
            clean_line=removeACLCitations(clean_line)
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
    cp.useLocalCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac")

    reader=AtharCorpusReader()
    all_contexts=[]
    all_docs=[]
    for f in glob.glob(infiles):
        print("Loading ",f)
        docs,contexts=reader.readFile(f)
        all_contexts.extend(contexts)
        all_docs.extend(docs)

    doc_dict={}
    for doc in all_docs:
        doc_dict[doc.metadata["guid"]]=doc.data

    json.dump(doc_dict,file(outfile,"w"))

def loadProcessedContexts(infile):
    """
    """
    return json.load(file(infile,"r"))

def corpusStatistics(infile):
    """
        Prints out statistics about the annotated contexts
    """
    data=json.load(file(infile,"r"))
    stats={}
    num_files=len(data)
    sources=[]
    targets=[]

    sentiments={}
    unique_sentiments={}
    for context in data:
        for line in context["lines"]:
            sent=line["sentiment"]
            if not sent: sent="x"
            sentiments[sent]=sentiments.get(sent,0)+1
            for s in sent:
                unique_sentiments[s]=unique_sentiments.get(s,0)+1
        sources.append(context["doc_from"])
        targets.append(context["doc_to"])
    sources=set(sources)
    targets=set(targets)

    print("Unique source files: ", len(sources))
    print("Unique target files: ", len(targets))
    print("Total contexts: ", len(data))
    print("Total annotated sentences: ", sum([sentiments[s] for s in sentiments]))
    print("Sentiment:")
    for sent in sentiments:
        print(sent,": ",sentiments[sent])
    print("Unique entiment:")
    for sent in unique_sentiments:
        print(sent,": ",unique_sentiments[sent])

def main():
    drive="g"
    processAtharCorpus(drive+r":\NLP\PhD\citation_context\*.html", drive+":\NLP\PhD\citation_context\doc_dict.json",)
##    corpusStatistics(drive+r":\NLP\PhD\citation_context\all_contexts.json")

##    contexts=loadProcessedContexts(r"G:\NLP\PhD\citation_context\all_contexts.json")
##
##    for context in contexts[:10]:
##        print(getOutlinkContextAtharAnnotated(context))
##        print("\n")
##        print(getOutlinkContextAtharWindowOfWords(context, 50, 50))
##        print("\n")

    pass

if __name__ == '__main__':
    main()
