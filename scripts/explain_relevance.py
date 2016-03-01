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
import minerva.db.corpora as cp
from general_utils import writeFileText
from context_extract import tokenizeText

from collections import defaultdict

def tokenizeAndTag(text,glob):
    """
        Function to be passed to SciDoc.prettyPrintHTML that will mark each token individually
    """
    glob["_token_counter"]=glob.get("_token_counter",0)
    glob["_inverted_index"]=glob.get("_inverted_index",defaultdict(lambda:set()))
    tokens=tokenizeText(text)
    res=[]
    for token in tokens:
        token_low=token.lower()
        glob["_inverted_index"][token_low].add(glob["_token_counter"])
        res.append('<span class="token" id="t'+str(glob["_token_counter"])+'">'+token+'</span>')
        glob["_token_counter"]+=1
    return " ".join(res)

def referenceFormatting(text,glob,ref):
    """
    """
    match=cp.Corpus.matcher.matchReference(ref)
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


def computeWordOverlap(tokens1,tokens2,inverted_index):
    """
        Makes a list of overlapping words with a given document, returns
    """
    set1=set(tokens1)
    set2=set(tokens2)

    common_tokens=set1 & set2

    this_doc_tokens=[]
    for token in common_tokens:
        this_doc_tokens.extend(inverted_index.get(token,[]))

    return this_doc_tokens

def computeOverlapForAllReferences(doc, inverted_index):
    """
    """
    text1=doc.getFullDocumentText(True,False)
    tokens1=tokenizeText(text1)
    per_reference_tokens={}
    for ref in doc["references"]:
        match=cp.Corpus.matcher.matchReference(ref)
        if match:
            doc2=cp.Corpus.loadSciDoc(match["guid"])
            text2=doc2.getFullDocumentText(True,False)
            tokens2=tokenizeText(text2)
            per_reference_tokens[ref["id"]]=computeWordOverlap(text1,text2,inverted_index)

    return per_reference_tokens

def explainRelevance(guid):
    """
        Given a guid, it prepares its explainer document
    """
    doc=cp.Corpus.loadSciDoc(guid)
    html=doc.prettyPrintDocumentHTML(True,True,False,tokenizeAndTag, referenceFormatting)
    computeOverlapForAllReferences(doc,doc.glob["_inverted_index"])
    writeFileText(html,cp.Corpus.dir_output+guid+"_explain.html")

def main():
    docs=cp.Corpus.listPapers("num_in_collection_references > 10 order by num_in_collection_references desc")
    explainRelevance(docs[0])
    pass

if __name__ == '__main__':
    main()
