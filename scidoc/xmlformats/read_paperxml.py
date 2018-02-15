# PaperXML converting to SciDocJSON.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import os, glob, re, codecs, json
import six.moves.cPickle, random
from bs4 import BeautifulStoneSoup
from pybtex.database.input.bibtexml import Parser as BibTeXMLParser
from pybtex.database import BibliographyDataError

from .base_classes import BaseSciDocXMLReader

from proc.nlp_functions import sentenceSplit
from proc.general_utils import loadFileText, writeFileText, normalizeUnicode, normalizeTitle
from scidoc.scidoc import SciDoc
from scidoc.render_content import SciDocRenderer
import db.corpora as cp
from parscit import ParsCitClient

from scidoc.citation_utils import (annotateCitationsInSentence,
matchCitationWithReference, normalizeAuthor, CITATION_FORM,
guessNamesOfPlainTextAuthor, fixNumberCitationsXML, detectCitationStyle)

from scidoc.reference_formatting import formatReference

def debugAddMessage(doc,prop,msg):
    """
        Prints a message and adds it to the specified tag of a document
    """
    print(msg)   # uncomment
    doc[prop]=doc.data.get(prop,"")+msg

class PaperXMLReader(BaseSciDocXMLReader):
    """
        Reader class for Paper/NLM XML

        read()
    """
    def __init__(self, parscit_api_url="http://127.0.0.1:5000/parscit/"):
        self.parscit_client=ParsCitClient(parscit_api_url)

    def cleanUpPaperXML(self, xml):
        """
            Cleans up some messy stuff in PaperXML
        """
        xml=xml.replace(u"\xad","") # this is badly processed end-of-line hyphens
        xml=normalizeUnicode(xml)
        return xml

    def cleanUpReferencesText(self, reftext):
        """
            Same as above, but called only on references it takes less time
        """
        reftext=re.sub(r"Pro­ceedings","Proceedings",reftext)
        reftext=re.sub(r"[Ii]n\s?\s?[Pp]roceedings\s+?of\s+?","In Proceedings of ", reftext)
        reftext=re.sub(r"([Ii]n)([A-Z\d])",r"\1 \2", reftext)
        # Break reference lines after "In proceedings of [x]. Name, Name, date..."
##        reftext=re.sub(r"((ceedings\s+of.{4,40}|pages.{4,12})\.\s*)([A-Z][a-z]{1,14}\s+[A-Z].{1,14})",r"\1 \n\n \3",reftext)
        # add space after commas. Seems to affect conversion quality a lot
        reftext=re.sub(r"([,;\"])([^\s])",r"\1 \2",reftext)
        # make sure there's space after a full stop
        reftext=re.sub(r"([a-z\?\!]\.)([A-Z])",r"\1 \2",reftext)
        # make sure there's space after a date's dot, parenthesis, or lack of space
        reftext=re.sub(r"((?:19|20)[0-9][0-9][a-g]?|in\spress|to\sappear|forthcoming|submitted)(\)?\.?)([\w\&;\"\'])",r"\1\2 \3",reftext)
        # Normalize line breaks
        reftext=re.sub(r"\n\r?\s?\n\r?\s?\n\r?\s?\n\r?\s?", r"\n\n",reftext)
        # Break apart several references on the same line
        # using apa_author and apa_year_num
        reftext=re.sub(r"(\w{2,150}\.) ((?:(?:(?:de |von |van )?[A-Z][A-Za-z'`-]+, [A-Z]\.) (?:and )?)+\((?:(?:19|20)[0-9][0-9][a-g]?|in\spress|to\sappear|forthcoming|submitted)\))",r"\1 \n\n \2",reftext)
        # similar to above, different format.
        reftext=re.sub(r"(\w{2,150}\.)  ?((?:(?:(?:de |von |van )?[A-Z][A-Za-z'`-]+[ \.])+(?:and )?)+\s?\(?(?:(?:19|20)?[0-9][0-9][a-g]?|in\spress|to\sappear|forthcoming|submitted)\)?)",r"\1 \n\n \2",reftext)
##        ref_lines=re.split(r"\n\r?\n\r?",reftext)
##        print(reftext)
        return reftext


    def loadPaperMainAuthorXML(self, author_node):
        """
            Returns BibJSON-compatible author info

            Args:
                author_node: XML node
            Returns:
                dict with author data
        """
        res={"family":author_node.get("surname",""),"given":author_node.get("givenname","")}
        orgs=author_node.findAll("org")
        if orgs != []:
            res["affiliation"]=[]
            for org in orgs:
                location=",".join([org.get("city",""),org.get("country","")]).strip(",")
                res["affiliation"].append({"name":org.get("name",""),"location":location})
        res=normalizeAuthor(res)
        return res

    def loadPaperMetadata(self, newDocument, soup, filename):
        """
            Tries to recover metadata from Paper file
        """
        header=soup.find("firstpageheader")
        if header:
            title=header.find("title")
            if title:
                newDocument.metadata["title"]=title.text

        path,fname=os.path.split(filename)
        metafilename=re.sub(r"(.*)-paper.xml",r"\1.xml",fname,flags=re.IGNORECASE)
        metafilename=os.path.join(path, metafilename)

        self.bibtex_parser = BibTeXMLParser()
##        print("trying to load BibTeXML from ", metafilename)
        try:
            bib_data = self.bibtex_parser.parse_file(metafilename)
        except BibliographyDataError as e:
            print(e)
        except:
            print("COULDN'T LOAD BIBTEXML FOR ",metafilename)
            bib_data=None

        if bib_data:
            entry=bib_data.entries[list(bib_data.entries.keys())[0]]
            for field in entry.fields:
                newDocument.metadata[field]=entry.fields[field].replace(u"\u2013",u"-")

        authors=[]
        for a in header.findChildren("author"):
            authors.append(self.loadPaperMainAuthorXML(a))
        newDocument["metadata"]["authors"]=authors
        newDocument["metadata"]["surnames"]=[a["family"] for a in authors]
        newDocument["metadata"]["norm_title"]=normalizeTitle(newDocument["metadata"]["title"])
##        print (json.dumps(newDocument.metadata),"\n\n")

    def loadPaperAbstract(self, soup, newDocument):
        """
            Loads the abstract, including sections
        """
        abstract_node=soup.find("abstract")
        if not abstract_node:
            debugAddMessage(newDocument,"error","CANNOT LOAD ABSTRACT! file: %s\n" % newDocument.metadata.get("filename","None"))
            # !TODO: LOAD first paragraph as abstract if no abstract available?
        else:
            abstract=newDocument.addSection("root","Abstract")

            paras=abstract_node.findAll("p")
            if len(paras) == 0:
                paras.append(abstract)
            for p in paras:
                self.loadPaperParagraph(p,newDocument,abstract["id"])

            newDocument.abstract=abstract

    def loadPaperSection(self, sec, newDocument, parent):
        """
            Gets called for each section.

            Args:
                sec: XML node
                newDocument: SciDoc
                parent: id of this section's parent in newDocument
        """
        header_id=0 # CHANGE
        header_text=sec.get("title","")
        # make sure first letter is capitalized
        if len(header_text) > 0:
            header_text=header_text[0].upper()+header_text[1:]

        newSection=newDocument.addSection(parent, header_text, header_id)

        contents=sec.findChildren(["subsection", "p", "figure"], recursive=False)
        if contents:
            for element in contents:
                if element.name=="subsection":
                    self.loadPaperSection(element,newDocument,newSection["id"])
                elif element.name=="p":
                    newPar=self.loadPaperParagraph(element, newDocument, newSection["id"])
                elif element.name=="figure":
                    newPar=newDocument.addParagraph(newSection["id"])
                    newSent=newDocument.addSentence(newPar["id"],"")
                    newSent["text"]=element.get("caption","")
                    newSent["type"]="fig-caption"
                    # TODO improve figure loading

    def loadPaperSentence(self, s, newDocument, parent):
        """
            Given a string, adds the sentence to the SciDoc, parses the citations,
            matches them with the references

            Args:
                s: string
                newDocument: SciDoc
                parent: id of element this sentence will hang from (p)
        """

        def replaceTempCitToken(s, temp, final):
            """
                replace temporary citation placeholder with permanent one
            """
            return re.sub(CITATION_FORM % temp, CITATION_FORM % final, annotated_s, flags=re.IGNORECASE)

        newSent=newDocument.addSentence(parent,"")

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
                newCit=newDocument.addCitation(sent_id=newSent["id"])
                # TODO check this: maybe not this simple. May need matching function.
                newCit["ref_id"]="ref"+str(int(citation["num"])-1)

                annotated_citations.append(newCit)
                annotated_s=replaceTempCitToken(annotated_s, index+1, newCit["id"])

        newSent["citations"]=[acit["id"] for acit in annotated_citations]
        newSent["text"]=annotated_s

        # deal with many citations within characters of each other: make them know they are a cluster
        # TODO cluster citations? Store them in some other way?
        newDocument.countMultiCitations(newSent)


    def loadPaperParagraph(self, p, newDocument, parent_id):
        """
            Creates a paragraph in newDocument, splits the text into sentences,
            creates a sentence object for each
        """
        if p.parent.name == "td":
            # This is not a content paragraph, but the content of a table cell
            return None

        par_text=p.renderContents(encoding=None)
        if re.match(r"(<i>)?proceedings\s+of\s+the\s+.*",par_text,flags=re.IGNORECASE):
            # This is not a content paragraph, we throw it away
            return None

        newPar=newDocument.addParagraph(parent_id)

        try:
            sentences=sentenceSplit(par_text)
        except:
            print("UNICODE ERROR!",par_text)
            sentences=[par_text]

        for s in sentences:
            self.loadPaperSentence(s,newDocument,newPar["id"])
        return newPar

    def loadPaperReferences(self, ref_section, doc):
        """
            Load the reference section

            Args:
                ref: xml node for reference element
                doc: SciDoc instance we're loading this for
            Returns: dict with the new loaded reference
        """

        all_elements=ref_section.findAll(["p","doubt"])
        process_elements=[]
        for index,element in enumerate(all_elements):
            if element.name=="doubt":
                if len(all_elements) > index+1:
                    all_elements[1].setString(element.text+" "+all_elements[1].text)
                elif index > 0:
                    all_elements[index-1].setString(all_elements[index-1].text+ " " + element.text)

        plain_text=[]
        for element in ref_section.findAll(["p"]):
            text=element.text
            plain_text.append(re.sub(r"</?i>"," ",text))


        reftext="\n\n".join(plain_text)
        # clean up the terrible references text, normalize spaces, commas, etc.
        reftext=self.cleanUpReferencesText(reftext)

        if plain_text == []:
            print("WARNING: NO REFERENCES! in ", doc.metadata.get("filename",""))
        else:
            parsed_refs=self.parscit_client.extractReferenceList(reftext)
            if parsed_refs:
                for ref in parsed_refs:
                    doc.addExistingReference(ref)
            else:
                raise ValueError("Couldn't parse references! in %s" % doc.metadata.get("filename",""))
                #TODO integrate FreeCite/others
                pass

##        print (json.dumps(doc.references))

    def read(self, xml, identifier):
        """
            Load a PaperXML into a SciDoc.

            Args:
                xml: full xml string
                identifier: an identifier for this document, e.g. file name
                        Important: supply an actual path so that we can check
                        for the meatadata in bibtexml
                        If an actual full path, the path will be removed from it
                        when stored
            Returns:
                SciDoc instance
        """
##        # this solves a "bug" in BeautifulStoneSoup with "sec" tags
##        BeautifulStoneSoup.NESTABLE_TAGS["sec"]=[]

        xml=self.cleanUpPaperXML(xml)
        soup=BeautifulStoneSoup(xml, convertEntities=BeautifulStoneSoup.HTML_ENTITIES)

        # Create a new SciDoc to store the paper
        newDocument=SciDoc()
        metadata=newDocument["metadata"]
        metadata["filename"]=os.path.basename(identifier)
##        if not citation_style:
##            raise ValueError("Cannot determine citation style")
            # default citation style if not otherwise detected
##            citation_style="APA"
        body=soup.find("body")
        if not body:
            # TODO: Make the error handling less terrible
            debugAddMessage(newDocument,"error","NO <BODY> IN THIS PAPER! file: "+identifier)
##            newDocument["metadata"]["guid"]=cp.Corpus.getFileUID(metadata["filename"])
            return newDocument


        # Load metadata, either from corpus or from file
        self.loadPaperMetadata(newDocument, soup, identifier)
        if metadata["surnames"] == []:
            debugAddMessage(newDocument,"error","NO SURNAMES OF AUTHORS file: "+identifier)
            return newDocument

        if metadata["title"] == []:
            debugAddMessage(newDocument,"error","NO TITLE file: "+identifier)
            return newDocument

        metadata["guid"]=cp.Corpus.generateGUID(metadata)

        # Load all references from the XML
        references=body.find("references")
        if references:
            self.loadPaperReferences(references, newDocument)

        newDocument.updateReferences()
##        print (newDocument.references)
##        print("\n\n")
        sections=body.findChildren("section", recursive=False)

        detect_style_text="".join([sec.renderContents() for sec in sections[:3]])
##        citation_style=detectCitationStyle(detect_style_text, default="APA")
        # turns out I don't have a good detection algorithm for AFI
        citation_style="APA"
        metadata["original_citation_style"]=citation_style

        # Load Abstract
        self.loadPaperAbstract(soup,newDocument)

        for sec in sections:
            self.loadPaperSection(sec, newDocument, "root")

        newDocument.updateReferences()
        newDocument.updateAuthorsAffiliations()
        return newDocument


def generateSideBySide(doc_list):
    """
        Generates side-by-side visualizations of a Paper XML: one using an XML to HTML
        converter, one loading the XML into SciDocJSON and rendering it back as HTML
    """
    from subprocess import Popen

    reader=PaperXMLReader()
    output_dir="g:\\nlp\\phd\\aac\\conversion_visualization\\"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list=[]
    for filename in doc_list:
        print("Converting %s" % filename)
        input_file=cp.Corpus.paths.inputXML+filename
        output_file=output_dir+"%s_1.html" % os.path.basename(filename)

        input_text=loadFileText(input_file)
        writeFileText(input_text,output_file)

        doc=reader.read(input_text, input_file)
        try:
            json.dumps(doc.data)
        except:
            print("Not JSON Serializable!!!!")

        html=SciDocRenderer(doc).prettyPrintDocumentHTML(True,True,True, True)
        output_file2=output_file.replace("_1.html","_2.html")
        writeFileText(html,output_file2)
        file_list.append([os.path.basename(output_file),os.path.basename(output_file2)])

    file_list_json="file_data=%s;" % json.dumps(file_list)
    writeFileText(file_list_json,output_dir+"file_data.json")


def inspectFiles(doc_list):
    """
        Goes file by file printing something about it
    """

    reader=PaperXMLReader()
    for filename in doc_list:
        input_file=os.path.join(cp.Corpus.dir_inputXML,filename)
        doc=reader.readFile(input_file)
        print(doc.data)
##        f=open(input_file)
##        lines=f.readlines()
##        soup=BeautifulStoneSoup("".join(lines))
##        elements=soup.findAll("org")
##        for element in elements:
##            print (element)

def basicTest():
    """
    """
    import db.corpora as cp

    drive="g"
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus(drive+":\\nlp\\phd\\aac")
##    doc_list=cp.Corpus.selectRandomInputFiles(3,"*-paper.xml")

    doc_list=[r"anthology\W\W10-0402-paper.xml"]
##    doc_list=[r"anthology\W\W11-2166-paper.xml"]

    generateSideBySide(doc_list)
##    inspectFiles(doc_list)
def main():
    basicTest()
##    reader=PaperXMLReader()
##    print(reader.cleanUpReferencesText("Yangarber, Roman and Ralph Grishman. 1998.&amp;quot;NYU: Description of the Proteus/PET System as Used for MUC-7 ST&amp;quot;In Proceedings of the 6thMessage Understanding Conference (MUC-7)"))
##    pass


if __name__ == '__main__':
    main()

