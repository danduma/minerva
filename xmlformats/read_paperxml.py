# PaperXML converting to SciDocJSON.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, glob, re, codecs, json
import cPickle, random
from BeautifulSoup import BeautifulStoneSoup
from pybtex.database.input.bibtexml import Parser as BibTeXMLParser
from pybtex.database import BibliographyDataError

from base_classes import BaseSciDocXMLReader
from citation_utils import guessNamesOfPlainTextAuthor, fixNumberCitationsXML, detectCitationStyle
from minerva.util.nlp_functions import sentenceSplit
from minerva.util.general_utils import loadFileText, writeFileText, normalizeUnicode, normalizeTitle
from minerva.scidoc import SciDoc, SciDocRenderer
import minerva.db.corpora as corpora
from minerva.parscit import ParsCitClient

from citation_utils import annotateCitationsInSentence, matchCitationWithReference, CITATION_FORM

from minerva.scidoc.reference_formatting import formatReference

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

# ------------------------------------------------------------------------------
#   Helper functions
# ------------------------------------------------------------------------------
    def cleanUpPaperXML(self, xml):
        """
        """
        xml=xml.replace(u"\xad","") # this is badly processed end-of-line hyphens
        xml=re.sub(r"Pro­ceedings","Proceedings",xml)
        xml=re.sub(r"In\s+?[Pp]roceedings\s+?of\s+?","In Proceedings of ", xml)
        xml=normalizeUnicode(xml)
        return xml

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
        print("trying to load BibTeXML from ", metafilename)
        try:
            bib_data = self.bibtex_parser.parse_file(metafilename)
        except BibliographyDataError as e:
            print(e)
        except:
            print("COULDN'T LOAD BIBTEXML FOR ",metafilename)
            bib_data=None

        if bib_data:
            entry=bib_data.entries[bib_data.entries.keys()[0]]
            for field in entry.fields:
                newDocument.metadata[field]=entry.fields[field]

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
        abstract=soup.find("abstract")
        if not abstract:
            debugAddMessage(newDocument,"error","CANNOT LOAD ABSTRACT! file: %s\n" % newDocument.metadata.get("filename","None"))
            # TODO: LOAD first paragraph as abstract if no abstract available
        else:
            abstract_id=newDocument.addSection("root","Abstract")

            paras=abstract.findAll("p")
            if len(paras) == 0:
                paras.append(abstract)
            for p in paras:
                self.loadPaperParagraph(p,newDocument,abstract_id)

            newDocument.abstract=newDocument.element_by_id[abstract_id]

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
        header_text=header_text[0].upper()+header_text[1:]

        newSection_id=newDocument.addSection(parent, header_text, header_id)

        contents=sec.findChildren(["subsection", "p", "figure"], recursive=False)
        if contents:
            for element in contents:
                if element.name=="subsection":
                    self.loadPaperSection(element,newDocument,newSection_id)
                elif element.name=="p":
                    newPar_id=self.loadPaperParagraph(element, newDocument, newSection_id)
                elif element.name=="figure":
                    newPar_id=newDocument.addParagraph(newSection_id)
                    newSent_id=newDocument.addSentence(newPar_id,"")
                    newSent=newDocument.element_by_id[newSent_id]
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

        newSent_id=newDocument.addSentence(parent,"")
        newSent=newDocument.element_by_id[newSent_id]

        annotated_s,citations_found=annotateCitationsInSentence(s, newDocument.metadata["original_citation_style"])

        annotated_citations=[]

        if newDocument.metadata["original_citation_style"]=="APA":
            for index,citation in enumerate(citations_found):
                newCit=newDocument.addCitation()
                newCit["parent"]=newSent_id
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
                newCit=newDocument.addCitation()
                newCit["parent"]=newSent_id
                # TODO check this: probably not this simple. Need matching function.
                newCit["ref_id"]=citation["num"]

                annotated_citations.append(newCit)
                annotated_s=replaceTempCitToken(annotated_s, index+1, newCit["id"])

        newSent["citations"]=[acit["id"] for acit in annotated_citations]
        newSent["text"]=annotated_s

        # deal with many citations within characters of each other: make them know they are a cluster
        # TODO cluster citations? Store them in some other way?
        newDocument.countMultiCitations(newSent)


    def loadPaperParagraph(self, p, newDocument, parent):
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

        newPar_id=newDocument.addParagraph(parent)

        try:
            sentences=sentenceSplit(par_text)
        except:
            print("UNICODE ERROR!",par_text)
            sentences=[par_text]

        for s in sentences:
            self.loadPaperSentence(s,newDocument,newPar_id)
        return newPar_id

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
        reftext=re.sub(r"((ceedings\s+of.{4,40}|pages.{4,12})\.\s*)([A-Z][a-z]{1,14}\s+[A-Z].{1,14})",r"\1 \n\n \3",reftext)
        reftext=reftext.replace("\n\r","")
        reftext=re.sub(r"\n\r?\s?\n\r?\s?\n\r?\s?\n\r?\s?", r"\n\n",reftext)
##        print(reftext)

        if plain_text == []:
            print("WARNING: NO REFERENCES! in ", doc.metadata["filename"])
        else:
            parsed_refs=self.parscit_client.extractReferenceList([reftext])[0]
            for ref in parsed_refs:
                doc.addExistingReference(ref)

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
        soup=BeautifulStoneSoup(xml)

        # Create a new SciDoc to store the paper
        newDocument=SciDoc()
        metadata=newDocument["metadata"]
        metadata["filename"]=os.path.basename(identifier)
        metadata["original_citation_style"]=detectCitationStyle(xml)

        body=soup.find("body")
        if not body:
            # TODO: Make the error handling less terrible
            debugAddMessage(newDocument,"error","NO <BODY> IN THIS PAPER! file: "+identifier)
            newDocument["metadata"]["guid"]=corpora.Corpus.getFileUID(metadata["filename"])
            return newDocument

        # Load metadata, either from corpus or from file
        self.loadPaperMetadata(newDocument, soup, identifier)
        metadata["guid"]=corpora.Corpus.generateGUID(metadata)

        # Load all references from the XML
        references=body.find("references")
        if references:
            self.loadPaperReferences(references, newDocument)

        newDocument.updateReferences()

        # Load Abstract
        self.loadPaperAbstract(soup,newDocument)

        for sec in body.findChildren("section", recursive=False):
            self.loadPaperSection(sec, newDocument, "root")

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
        input_file=corpora.Corpus.dir_inputXML+filename
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
        input_file=os.path.join(corpora.Corpus.dir_inputXML,filename)
        doc=reader.readFile(input_file)
        print(doc.data)
##        f=open(input_file)
##        lines=f.readlines()
##        soup=BeautifulStoneSoup("".join(lines))
##        elements=soup.findAll("org")
##        for element in elements:
##            print (element)

def main():
    import minerva.db.corpora as corpora
    drive="g"
    corpora.Corpus.connectCorpus(drive+":\\nlp\\phd\\aac")
##    doc_list=corpora.Corpus.selectRandomInputFiles(3,"*-paper.xml")

##    doc_list=[r"g:\nlp\phd\aac\inputXML\anthology\W\W10-0402-paper.xml"]
    doc_list=[r"anthology\W\W11-2166-paper.xml"]

    generateSideBySide(doc_list)
##    inspectFiles(doc_list)

    pass


if __name__ == '__main__':
    main()

