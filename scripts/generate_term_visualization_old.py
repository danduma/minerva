#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     generates the data to visualize overlaps of terms between citing
#              document and cited document
#
# Author:      dd
#
# Created:     22/04/2015
# Copyright:   (c) dd 2015
#-------------------------------------------------------------------------------


from pandas import *
from corpora import Corpus
from general_utils import writeFileText
from context_extract import tokenizeText, getDictOfTokenCounts
import nlp_functions

import json, re

from collections import defaultdict

local_stopwords_list=[str(num) for num in range(100)]
local_stopwords_list.extend(["abstract","introduction"])
local_stopwords_list.extend(nlp_functions.stopwords_list)
local_stopwords_list.extend(nlp_functions.stopwords)

class VisGenerator:
    """
        Class wrapper over functinos that generate the visualization data
    """
    def __init__(self):
        self.term_info={}

    def getDocumentTokens(self, doc):
        """
            Returns a dict of all terms in the document
        """
        full_text=doc.getFullDocumentText(True,False)
        full_text=full_text.lower()
        tokens=tokenizeText(full_text)
        # remove many stopwords. hack!
        tokens=[token for token in tokens
            if token not in local_stopwords_list]
        counts=getDictOfTokenCounts(tokens)
        return counts

    def getOverlappingTokens(self, counts1, counts2):
        """
            Returns the intersection of the sets of counts1 and counts2
        """
        return list(set(counts1.keys()) & set(counts2.keys()))

    def textFormatting(self, text, glob):
        """
            Text formatting function to be passed to JsonDoc.prettyPrintDocumentHTML()

            Args:
                text: text of sentence
                glob: globals dictionary
        """
        res=[]
        text=text.strip(".")
        tokens=tokenizeText(text)
        for token in tokens:
            token_dict=self.term_info.get(token.lower(),
                {"token_id":0, "references":[]})
            references=" ".join(token_dict["references"])
            classes=str(token_dict["token_id"]) + " " + references
            res.append('<span term_id="%s" class="%s">%s</span>' %
                (str(token_dict["token_id"]),classes,token))

        result=" ".join(res).strip()
        return result.strip(".")

    def extraAttributes(self, s,attributes, glob):
        """

        """
        if s.get("type","") != "s":
            return attributes

        attributes["class"]=attributes.get("class","")+" AZ_"+s["az"]
        attributes["class"]+=" CSC_"+s.get("csc_type","Bac")
        attributes["class"]=attributes["class"].strip()

    def referenceFormatting(self, text, glob, ref):
        """
            Just adds a span around each reference, distinguishing between in-collection
            references and not. Behaviour depends on CSS/JS

            Function to be passed to JsonDoc.prettyPrintDocumentHTML()

            Args:
                text: text of reference
                glob: globals dictionary
                ref: reference dictionary
        """
        reftype="reference"
        if ref["in_collection"]:
            reftype+=" in-collection_ref"
        res="<span class='%s' id='%s'>%s</span>" % (reftype, ref["id"], text)
        return res

    def citationFormatting(self, formatted_citation, reference):
        """
            Add extra formatting to an already formatted citation element, using
            info from reference
        """
        extra_class="in-collection_cit " if reference["in_collection"] else ""
        res=re.sub(r"class=\"","class=\""+extra_class, formatted_citation, flags=re.IGNORECASE)
        return res

    def padWithHTML(self, html, guid):
        """
            Adds header libraries and such to HTML
        """
        result="""
        <html>
        <head>
        <link href='visualization.css' rel='stylesheet'/>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script src="%s_data.json"></script>
        <script src="visualization.js"></script>
        </head>
        <body onload="setupHandlers()">
        %s
        </body></html>
        """ % (guid, html)
        return result

    def filterTokens(self):
        """
            Removes tokens that are too common (stopwords) or shared by too many
            references
        """
        to_remove=[]
        # more than this many references contain this word, cut it out
        cutoff= len(doc["references"]) * 0.4
##        cutoff= len(doc["references"]) * 1
        for token in self.term_info:
            if len(self.term_info[token]["references"]) > cutoff:
                to_remove.append(token)

        for token in to_remove:
            del self.term_info[token]

    def generateVisualization(self, guid):
        """
            Given a guid, it prepares its explainer document
        """
##        output_dir=Corpus.dir_output
        output_dir=r"C:\Users\dd\Documents\Dropbox\PhD\code\doc_visualization\\"

        doc=Corpus.loadSciDoc(guid)
        Corpus.tagAllReferencesAsInCollectionOrNot(doc)
        counts1=self.getDocumentTokens(doc)

        # generate a unique id for each unique term, make a dictionary
        for index, token in enumerate(counts1):
            self.term_info[token]={"token_id":str(index), "references": []}

        self.overlapping_tokens={}

        in_collection_references=Corpus.getMetadataByGUID(guid)["outlinks"]
        for ref in doc["references"]:
            match=Corpus.matchReferenceInIndex(ref)
            if match:
                doc2=Corpus.loadSciDoc(match["guid"])
                counts2=self.getDocumentTokens(doc2)
                # for each in_collection_reference number (0 onwards) we store the list
                # of its overlapping tokens with the current document

                self.overlapping_tokens[ref["id"]]=self.getOverlappingTokens(counts1, counts2)

                for token in self.overlapping_tokens[ref["id"]]:
                    ref_list=self.term_info[token]["references"]
                    if ref["id"] not in ref_list:
                        ref_list.append(ref["id"])

        # try to find some signal in the noise
        self.filterTokens()

        json_str="var token_data=" + json.dumps(self.overlapping_tokens) + ";"
        writeFileText(json_str, output_dir+guid+"_data.json")

        html=doc.prettyPrintDocumentHTML(
            True,
            True,
            False,
##            extra_attribute_function=self.extraAttributes,
            citation_formatting_function=self.citationFormatting,
            reference_formatting_function=self.referenceFormatting,
            text_formatting_function=self.textFormatting
            )
        html=self.padWithHTML(html, guid)
        writeFileText(html,output_dir+guid+"_vis.html")

from augment_scidocs import augmentSciDocWithAZ

def main():
##    explainZoning("P95-1026")

    docs=Corpus.listPapers("num_in_collection_references > 10 order by num_in_collection_references desc")
    generator=VisGenerator()
    generator.generateVisualization(docs[0])

##    meta=Corpus.getMetadataByGUID("P95-1026")
##    explainZoning("P95-1026")
##    for link in meta["inlinks"]:
##        print link
####        augmentSciDocWithAZ(link)
##        explainZoning(link)
##    pass

if __name__ == '__main__':
    main()
