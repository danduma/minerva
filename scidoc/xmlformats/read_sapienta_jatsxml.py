# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import re

from .read_jatsxml import JATSXMLReader
from bs4 import BeautifulStoneSoup

from scidoc.citation_utils import (annotateCitationsInSentence,
CITATION_FORM, matchCitationWithReference)

class SapientaJATSXMLReader(JATSXMLReader):
    """
        Reader class for JATS/NLM XML

        Main entry point: read()
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        # Need to create new IDs for references because we're annotating references again
        self.USE_ORIGINAL_REF_ID=False

    def loadJATSSectionTitle(self, sec):
        """
            Loads a section's title.

            :param sec: section XML node
        """
        header_text=""
        title=sec.find("title", recursive=False)
        if not title:
            header_id=0
        else:
            header_id=0 # CHANGE
            text=title.find("text", recursive=False)
            if text:
                s=text.find("s",recursive=False)
                if s:
                    plain=s.find("plain")
                    if plain:
                        header_text=re.sub(r"</?title.*?>","",plain.text)
                        # make sure first letter is capitalized
                        try:
                            header_text=header_text[0].upper()+header_text[1:]
                        except:
                            header_text=""
            else:
                print(("Weird, title tag is there but no text is to be found: %s" % (title)))

        return header_text, header_id

    def loadJATSParagraph(self, p, newDocument, parent):
        """
            Creates a paragraph in newDocument, splits the text into sentences,
            creates a sentence object for each
        """
        newPar=newDocument.addParagraph(parent)
        texts=p.findChildren("text", recursive=False)
        if texts:
            for text in texts:
                sent_tags=text.findChildren("s",recursive=True)
                for s in sent_tags:
                    self.loadJATSSentence(s, newDocument, newPar["id"], parent)
        else:
            pass
##            par_text=p.renderContents(encoding=None)
##
##            sentences=sentenceSplit(par_text)
##            for s in sentences:
##                self.loadJATSSentence(s, newDocument, newPar["id"], parent)

        return newPar["id"]

    def annotatePlainTextCitations(self, s, newDocument, newSent):
        """
            If the citations aren't tagged with <xref> because Sapienta stripped
            them away, try to extract the citations from plain text. *sigh*

            :param s: BeautifulSoup tag of the sentence
            :param newDocument: SciDoc instance we are populating
            :param newSent: new sentence in this document that we are adding
        """

        def replaceTempCitToken(s, temp, final):
            """
                replace temporary citation placeholder with permanent one
            """
            return re.sub(CITATION_FORM % temp, CITATION_FORM % final, annotated_s, flags=re.IGNORECASE)

        def replaceTempCitTokenMulti(s, temp, final_list):
            """
                Replace temporary citation placeholder with a list of permanent
                ones to deal with multi citations, e.g. [1,2,3]
            """
            assert(isinstance(final_list, list))
            rep_string="".join([CITATION_FORM % final for final in final_list])
            return re.sub(CITATION_FORM % temp, rep_string, annotated_s, flags=re.IGNORECASE)

        if not newDocument.metadata.get("original_citation_style", None):
            newDocument.metadata["original_citation_style"]="AFI"
        annotated_s,citations_found=annotateCitationsInSentence(s, newDocument.metadata["original_citation_style"])
        annotated_citations=[]

        if newDocument.metadata["original_citation_style"]=="APA":
            for index,citation in enumerate(citations_found):
                newCit=newDocument.addCitation(sent_id=newSent["id"])
                reference=matchCitationWithReference(citation, newDocument["references"])
##                print (citation["text"]," -> ", formatReference(reference))
                if reference:
                    newCit["ref_id"]=reference["id"]
                else:
                    # do something else?
                    newCit["ref_id"]=None
                annotated_citations.append(newCit)
                annotated_s=replaceTempCitToken(annotated_s, index+1, newCit["id"])

        elif newDocument.metadata["original_citation_style"]=="AFI":
            for index,citation in enumerate(citations_found):
                valid_citation=True
                nums=[]
                for num in citation["nums"]:
                    cit_num=int(num)-1
                    if cit_num < 0:
                        # this is not a citation! Probably something like "then sampled a random number from uniform distribution, u ~ U[0,1]"
                        valid_citation=False
                        break
                    nums.append(cit_num)

                if not valid_citation:
                    continue

                cit_ids=[]
                for num in nums:
                    newCit=newDocument.addCitation(sent_id=newSent["id"])
                    # TODO check this: maybe not this simple? May need matching function.
                    newCit["ref_id"]="ref"+str(num)
                    cit_ids.append(newCit["id"])
                    annotated_citations.append(newCit)

                annotated_s=replaceTempCitTokenMulti(annotated_s, index+1, cit_ids)

        if len(annotated_citations) > 0:
            newSent["citations"]=[acit["id"] for acit in annotated_citations]
        newSent["text"]=annotated_s

    def loadJATSSentence(self, s, newDocument, par_id, section_id):
        """
            Loads a Sapienta-tagged JATS sentence

            :param s_soup: the <s> tag from BeautifulStoneSoup
            :param newDocument: SciDoc
            :type newDocument: SciDoc
            :param par_id: id of the paragraph containing this sentence
            :type par_id: string
            :param section_id: id of the section containing the paragraph
        """
        s_soup=s.find("plain",recursive=True)
        if s_soup is None:
            # <s> tag that contains no actual text. We can return without adding any sentence
            return

        newSent=newDocument.addSentence(par_id,"")
        coresc_tag=s.find("coresc1",recursive=False)
        newSent["csc_type"]=coresc_tag["type"]
        newSent["csc_adv"]=coresc_tag["advantage"]
        newSent["csc_nov"]=coresc_tag["novelty"]

        refs=s_soup.findAll("xref",{"ref-type":"bibr"})
        citations_found=[]
        for r in refs:
            citations_found.extend(self.loadJATSCitation(r, newSent["id"], newDocument, section=section_id))

        non_refs=s_soup.findAll(lambda tag:tag.name.lower()=="xref" and "ref-type" in tag and tag["ref-type"].lower() != "bibr")
        for nr in non_refs:
            nr.name="inref"

        if len(citations_found) > 0:
            newSent["citations"]=[acit["id"] for acit in citations_found]
            # TODO replace <xref> tags with <cit> tags
            newSent["text"]=newDocument.extractSentenceTextWithCitationTokens(s_soup, newSent["id"])
        else:
            self.annotatePlainTextCitations(s_soup.text,  newDocument, newSent)

##            print(newSent["text"])
        # deal with many citations within characters of each other: make them know they are a cluster
        # TODO cluster citations? Store them in some other way?
        newDocument.countMultiCitations(newSent)

    def read(self, xml, identifier):
        """
            Load a Sapienta-annotated JATS/NLM (PubMed) XML into a SciDoc.

            :param xml: full xml string
            :type xml: basestring
            :param identifier: an identifier for this document, e.g. file name
                        If an actual full path, the path will be removed from it
                        when stored
            :type identifier: basestring
            :returns: :class:`SciDoc <SciDoc>` object
            :rtype: SciDoc
        """
        # this solves a "bug" in BeautifulStoneSoup with "text" tags
        BeautifulStoneSoup.NESTABLE_TAGS["text"]=[]
        return super(self.__class__, self).read(xml, identifier)


def main():
    import db.corpora as cp

    drive="g"
    cp.useLocalCorpus()
    cp.Corpus.connectCorpus(drive+":\\nlp\\phd\\pmc_coresc\\")

    from . import read_jatsxml

    debugging_files=[
    r"data\scratch\mpx245\epmc\output\Out_PMC3184115_PMC3205799.xml.gz.gz\3187739_annotated.xml",
##        r"Out_PMC549041_PMC1240567.xml.gz.gz\555959_done.xml",
##        r"Out_PMC549041_PMC1240567.xml.gz.gz\555763_done.xml",
    ]

    read_jatsxml.generateSideBySide(debugging_files)
    pass

if __name__ == '__main__':
    main()
