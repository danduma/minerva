#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      dd
#
# Created:     22/04/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from pandas import *
from corpora import Corpus
from general_utils import writeFileText
from context_extract import tokenizeText

from collections import defaultdict

def extraAttributes(s,attributes, glob):
    """

    """
    if s.get("type","") != "s":
        return attributes

    attributes["class"]=attributes.get("class","")+" AZ_"+s["az"]
    attributes["class"]+=" CSC_"+s.get("csc_type","Bac")
    attributes["class"]=attributes["class"].strip()

def referenceFormatting(text,glob,ref):
    """
        Function to be passed to JsonDoc.prettyPrintDocumentHTML()

        Just adds a span around each reference, distinguishing between in-collection
        references and not. Behaviour depends on CSS/JS
    """
    match=Corpus.matchReferenceInIndex(ref)
    reftype="reference"
    if match:
        reftype+=" in-collection"
    res="<span class='"+reftype+"'>"+text+"</span>"
    return res

def padWithHTML(html):
    """
    """
    result="""<html><head><link href='scidocview.css' rel='stylesheet'>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
    </head><body>"""
    result+=html+"</body></html>"
    return result


def explainZoning(guid):
    """
        Given a guid, it prepares its explainer document
    """
    doc=Corpus.loadSciDoc(guid)
    html=doc.prettyPrintDocumentHTML(True,True,False, extra_attribute_function=extraAttributes, reference_formatting_function=referenceFormatting)
    html=padWithHTML(html)
    writeFileText(html,Corpus.dir_output+guid+"_zoning.html")

from augment_scidocs import augmentSciDocWithAZ

def main():
##    docs=Corpus.listPapers("num_in_collection_references > 10 order by num_in_collection_references desc")
##    explainZoning("P95-1026")
    meta=Corpus.getMetadataByGUID("P95-1026")
    explainZoning("P95-1026")
    for link in meta["inlinks"]:
        print link
##        augmentSciDocWithAZ(link)
        explainZoning(link)
    pass

if __name__ == '__main__':
    main()
