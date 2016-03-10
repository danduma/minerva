# JATS/NLM XML converting to SciDocJSON. See http://jats.nlm.nih.gov/
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from base_classes import BaseSciDocXMLReader
from minerva.scidoc.citation_utils import guessNamesOfPlainTextAuthor, fixNumberCitationsXML, detectCitationStyle

import os, glob, re, codecs, json, logging
import cPickle, random
from BeautifulSoup import BeautifulStoneSoup

from minerva.proc.nlp_functions import sentenceSplit
from minerva.proc.general_utils import (writeFileText, pathSelect, cleanxml)

from minerva.scidoc.scidoc import SciDoc
from minerva.scidoc.render_content import SciDocRenderer
import minerva.db.corpora as cp

def debugAddMessage(doc,prop,msg):
    """
        Prints a message and adds it to the specified tag of a document
    """
    print(msg)   # uncomment
    doc[prop]=doc.data.get(prop,"")+msg

class JATSXMLReader(BaseSciDocXMLReader):
    """
        Reader class for JATS/NLM XML

        Main entry point: read()
    """
    def __init__(self):
        self.USE_ORIGINAL_REF_ID=True
        pass

# ------------------------------------------------------------------------------
#   Helper functions
# ------------------------------------------------------------------------------
    def loadValueFromXMLifExists(self, node,value,default):
        """
            If node exists and node has a subnode called value, return node.value.text,
            else return default
        """
        if not node: return default
        subnode=node.find(value)
        if subnode:
            return subnode.text
        else:
            return default

    def extractInfoFromPatentText(self, text, newRef):
        """
            Read patent data from XML.

            :param text: the text content of the reference
            :type text: string
            :param newRef: the already added reference dictionary
            :returns: True if patent info was found, False otherwise
            :rtype: boolean
        """
        text=re.sub(r"<ext-link.*?</ext-link>"," ",text,0,re.IGNORECASE)
        text=re.sub(r"Accessed\s.*?\d{4}"," ",text,0,re.IGNORECASE)
        rpatent=r"<mixed-citation.*?>(.+?),\s?inventor.*?assignee\.(.+?)\..+?patent\s+?(.+?)\.\s+?(.*?)\.\s+?<"
        match=re.search(rpatent,text,re.IGNORECASE | re.DOTALL)
        if match:
            newRef["publication-type"]="patent"
            newRef["authors"]=match.group(1).split(",")
            newRef["title"]=match.group(2)
            newRef["patent_num"]=match.group(3)
            m2=rxsingleyear.search(match.group(3))
            if m2:
                newRef["year"]=m2.group(0)
            return True
        else:
            rpatent=r"<mixed-citation.*?>(.+?)\.\s+(.+?)\.\s?(.*?)\s+(\d{4})\.<"
            match=re.search(rpatent,text,re.IGNORECASE | re.DOTALL)
            if match:
                newRef["publication-type"]="other"
                newRef["author"]=match.group(1)
                newRef["title"]=match.group(2)
                newRef["year"]=match.group(4)
                return True
        return False

    def loadJATSmetadataIfExists(self, element_node, items_to_load, result_dict):
        """
            This loads optional BibTeX stuff like issue, volume, fpage, lpage, etc.
        """
        for item in items_to_load:
            element=element_node.find(item)
            if element:
                result_dict[item]=element.text

    def selectBestDateFromMetadata(self, meta, newDocument):
        """
            Picks a date from the extremely detailed PubMed publication history.

            Args:
                meta: metadata dictionary
                newDocument: SciDoc instance we're doing this for
            Returns:
                pointer to the newDocument instance
        """
        pubs=meta.findAll("pub-date")
        all_pubs=[]
        for pub in pubs:
            pubdate={"pub-type":pub.get("pub-type","")}
            for item in["day","month","year"]:
                found=pub.find(item)
                if found:
                    pubdate[item]=found.text
            all_pubs.append(pubdate)

        newDocument["metadata"]["publication_history"]=all_pubs

        for priority in ["pmc-release","epub","ppub","pub",""]:
            for pub in all_pubs:
                if pub.get("pub-type","")==priority:
                    newDocument["metadata"]["year"]=pub["year"]
                    return newDocument

        return newDocument

    def loadJATSCitation(self, ref, sentence_id, newDocument, section):
        """
            Extract all info from <ref> tag, return dictionary

            Args:
                ref: <xref> tag leaf
                sentence_id: id of setnence containing citation
                newDocument: SciDoc
                section: section containing citation
        """
        citations_list=ref["rid"].split()
        res=[]

        for ref_id in citations_list:
            #TODO does this work?
            real_ref=newDocument.findMatchingReferenceByOriginalId(ref_id)
            real_ref_id=["id"] if real_ref else ref_id

            newCit=newDocument.addCitation(sentence_id, real_ref_id)

            if ref.has_key("citation_id"):
                res["original_id"]=ref["citation_id"]

            newCit["original_text"]=ref.__repr__()
            newCit["parent_section"]=section

            if not newDocument.reference_by_id.has_key(real_ref_id):
                # something strange is going on: no reference with that key
                print ("No reference found with id: %s" % real_ref_id)
    ##            continue
            else:
                newDocument.reference_by_id[real_ref_id]["citations"].append(newCit["id"])

            res.append(newCit)
        return res

    def loadJATSMainAuthorXML(self, author_node):
        """
            Returns BibJSON-compatible author info

            Args:
                author_node: XML node
            Returns:
                dict with author data
        """
        res={"family":"","given":""}
        name=author_node.find("name")

        if name:
            for item in [("surname","family"),("given-names","given"),("degrees","degrees")]:
                node=name.findChild(item[0],recursive=False)
                if node:
                    res[item[1]]=node.text

        xrefs=author_node.findAll("xref",{"ref-type":"aff"})
        affs=author_node.parent.parent.findAll("aff")
        for xref in xrefs:
            if xref.text.strip() != "":
                xref_id=xref.text
            else:
                xref_id=xref.get("rid", None)

            aff_id=""
            for aff in affs:
                label=aff.find("label")
                if label:
                    aff_id=label.text
                else:
                    if not aff.has_key("id"):
                        print ("Aff with no id: %s" % aff)
                    else:
                        aff_id=aff["id"]

                if aff_id==xref_id:
                    res["affiliation"]=res.get("affiliation",[])
                    for node in aff.findAll(True):
                        res["affiliation"].append({"type":node.name,"name":node.text})
        return res

    def loadMetadataIfExists(self, branch, key, doc):
        """
            If branch is not none, load its text into the metadata dict under the
            given key
        """
        if branch:
            doc.metadata[key]=branch.text

    def loadJATSMetadataFromPaper(self, newDocument, soup):
        """
            Tries to recover metadata from the header of the JATS file
        """
        front=soup.find("front")
        if not front:
            debugAddMessage(newDocument,"error","NO <front> IN DOCUMENT! file:"+filename)
            return newDocument

        journal_meta=front.find("journal-meta")
        if journal_meta:
            self.loadMetadataIfExists(journal_meta.find("journal-id",{"journal-id-type":"nlm-ta"}), "journal_nlm-ta", newDocument)
            self.loadMetadataIfExists(journal_meta.find("journal-id",{"journal-id-type":"iso-abbrev"}), "journal_iso-abbrev", newDocument)
            self.loadMetadataIfExists(journal_meta.find("journal-id",{"journal-id-type":"publisher-id"}), "publisher-id", newDocument)
            self.loadMetadataIfExists(journal_meta.find("publisher-name"), "publisher-name", newDocument)

        meta=front.find("article-meta")
        if meta:
            try:
                newDocument.metadata["title"]=meta.find("title-group").find("article-title").text
            except:
                newDocument.metadata["title"]="<NO TITLE>"

            self.loadMetadataIfExists(meta.find("article-id",{"pub-id-type":"pmid"}), "pm_id", newDocument)
            self.loadMetadataIfExists(meta.find("article-id",{"pub-id-type":"pmc"}), "pmc_id", newDocument)
            self.loadMetadataIfExists(meta.find("article-id",{"pub-id-type":"pmcid"}), "pmc_id", newDocument)
            self.loadMetadataIfExists(meta.find("article-id",{"pub-id-type":"doi"}), "doi", newDocument)
            self.loadMetadataIfExists(meta.find("article-id",{"pub-id-type":"publisher-id"}), "publisher-id", newDocument)

        authors=[]
        if not meta:
            debugAddMessage(newDocument,"error","NO METADATA IN DOCUMENT! file:"+filename)
            return newDocument


        for a in meta.findChildren("contrib", {"contrib-type":"author"},recursive=True):
            authors.append(self.loadJATSMainAuthorXML(a))

        self.loadJATSmetadataIfExists(meta, ["volume", "issue", "fpage", "lpage"],newDocument.metadata)

        permissions=meta.find("permissions")
        if permissions:
            self.loadJATSmetadataIfExists(permissions,["copyright-statement", "copyright-year"], newDocument.metadata)
            license=permissions.find("license")
            if license:
                newDocument.metadata["license-type"]=license.get("license-type","")
                newDocument.metadata["license-url"]=license.get("license-type","")

        newDocument["metadata"]["authors"]=authors
        newDocument["metadata"]["surnames"]=[a["family"] for a in authors]

        # load all publication dates and choose the "year"
        self.selectBestDateFromMetadata(meta, newDocument)
        if newDocument.metadata.get("pmc_id","") != "":
            newDocument.metadata["corpus_id"]=newDocument.metadata["pmc_id"]

##        print (json.dumps(newDocument.metadata),"\n\n\n\n\n")

    def loadJATSAbstract(self, soup, newDocument):
        """
            Loads the abstract, including sections

            :param soup: BeautifulStoneSoup we're loading from
            :param newDocument: SciDoc being populated
        """
        abstract=soup.find("abstract")
        if not abstract:
##            debugAddMessage(newDocument,"error","CANNOT LOAD ABSTRACT! file: %s\n" % newDocument.metadata.get("filename","None"))
            logging.warning("CANNOT LOAD ABSTRACT! file: %s\n" % newDocument.metadata.get("filename","None"))
            newDocument.metadata["missing_abstract"] = True
            # TODO: LOAD first paragraph as abstract if no abstract available
        else:
            newAbstract=newDocument.addSection("root","Abstract")

            abstract_sections=abstract.findAll("sec")
            if abstract_sections:
                for sec in abstract_sections:
                    title=sec.find("title")
                    title=sec.title.text if title else ""
                    newSection=newDocument.addSection(newAbstract["id"],title)
                    for p in sec.findAll("p"):
                        self.loadJATSParagraph(p,newDocument,newSection["id"])
            else:
                for p in abstract.findAll("p"):
                    self.loadJATSParagraph(p,newDocument,newAbstract["id"])

            newDocument.abstract=newDocument.element_by_id[newAbstract["id"]]

    def loadJATSSectionTitle(self, sec):
        """
            Loads a section's title.

            :param sec: section XML node
        """
        header=sec.find("title", recursive=False)
        if not header:
            header_id=0
            header_text=""
        else:
            header_id=0 # CHANGE
            header_text=re.sub(r"</?title.*?>","",header.__repr__())
            # make sure first letter is capitalized
            header_text=header_text[0].upper()+header_text[1:]

        return header, header_id

    def loadJATSSection(self, sec, newDocument, parent):
        """
            Gets called for each section.

            :param sec: section XML node
            :param newDocument: SciDoc
            :param parent: id of this section's parent in newDocument
        """
        header_text, header_id=self.loadJATSSectionTitle(sec)

        newSection=newDocument.addSection(parent, header_text, header_id)

        contents=sec.findChildren(["sec","p"],recursive=False)
        if contents:
            for element in contents:
                if element.name=="sec":
                    self.loadJATSSection(element,newDocument,newSection["id"])
                elif element.name=="p":
                    newPar_id=self.loadJATSParagraph(element, newDocument, newSection["id"])


    def loadJATSSentence(self, s, newDocument, par_id, section_id):
        """
            Loads a JATS sentence (ready split)

            :param s: the plain text of the sentence (with all tags inside, e.g. <xref>)
            :param newDocument: SciDoc
            :param par_id: id of the paragraph containing this sentence
            :param section_id: id of the section containing the paragraph
        """
        newSent=newDocument.addSentence(par_id,"")
        s_soup=BeautifulStoneSoup(s)

        refs=s_soup.findAll("xref",{"ref-type":"bibr"})
        citations_found=[]
        for r in refs:
            citations_found.extend(self.loadJATSCitation(r, newSent["id"], newDocument, section=section_id))

        non_refs=s_soup.findAll(lambda tag:tag.name.lower()=="xref" and tag.has_key("ref-type") and tag["ref-type"].lower() != "bibr")
        for nr in non_refs:
            nr.name="inref"

        newSent["citations"]=[acit["id"] for acit in citations_found]
        # TODO replace <xref> tags with <cit> tags
        newSent["text"]=newDocument.extractSentenceTextWithCitationTokens(s_soup, newSent["id"])
##            print(newSent["text"])
        # deal with many citations within characters of each other: make them know they are a cluster
        # TODO cluster citations? Store them in some other way?
        newDocument.countMultiCitations(newSent)

    def loadJATSParagraph(self, p,newDocument, parent):
        """
            Creates a paragraph in newDocument, splits the text into sentences,
            creates a sentence object for each
        """
        newPar=newDocument.addParagraph(parent)
        par_text=p.renderContents(encoding=None)
        sentences=sentenceSplit(par_text)
        for s in sentences:
            self.loadJATSSentence(s, newDocument, newPar["id"], parent)

        return newPar

    def loadJATSReference(self, ref, doc):
        """
            Load a reference from the bibliography section.

            :param ref: xml node for reference element
            :param doc: :class `SciDoc <SciDoc>` instance we're loading this for
            :returns: dict with the new loaded reference
            :rtype: dict
        """

        xmltext=ref.__repr__()
        authorlist=[]
        surnames=[]
        original_id=ref["id"]

        citation_type_key="publication-type"

        element=ref.find("element-citation")
        if not element:
            element=ref.find("mixed-citation")
            if not element:
                element=ref.find("citation")
                if element:
                    citation_type_key="citation-type"

        author_group=ref.find("person-group",{"person-group-type":"author"})
        if not author_group:
            collab=ref.find("collab")
            if collab:
                authorlist.append(guessNamesOfPlainTextAuthor(collab.text))
            else:
                author_group=ref

        if author_group:
            authors=author_group.findAll("name")
        else:
            authors=None
            collab=ref.find("collab")
            if collab:
                authorlist.append({"family":collab.text, "given":""})
                surnames.append(collab.text)

        if authors:
            for a in authors:
                astring=a.__repr__()
                surname=a.find("surname")
                if surname:
                    surnames.append(surname.text)
                given_names=a.find("given-names")
                if given_names and surname:
                    authorlist.append({"given": given_names.text, "family": surname.text})
                else:
                    astring=cleanxml(astring)
                    authorlist.append(guessNamesOfPlainTextAuthor(astring))
        else:
            srnms=ref.findAll("surname")
            for s in srnms:
                surnames.append(s.text)

        newref=doc.addReference()
##        newref["xml"]=xmltext
    ##    newref["text"]=cleanxml(xmltext)
        newref["authors"]=authorlist
        newref["surnames"]=surnames
        newref["external_links"]=[]
        newref["title"]="<NO TITLE>"
        if not element:
            newref["xml"]=xmltext
            return newref

        article_title=ref.find("article-title")
        source=element.find("source")
        if source:
            newref["publication-name"]=source.text
        else:
            newref["publication-name"]=""

        try:
            newref["publication-type"]=element[citation_type_key]
        except:
            newref["publication-type"]="unknown"

        if newref["publication-type"]=="book":
            if source:
                newref["title"]=source.text
            else:
                if article_title:
                    newref["title"]=article_title.text
        elif newref["publication-type"]=="journal":
            if article_title:
                newref["title"]=article_title.text
        elif newref["publication-type"]=="other":
            if article_title:
                newref["title"]=article_title.text
            elif source:
                newref["title"]=source.text
            self.extractInfoFromPatentText(ref.__repr__(), newref)

        self.loadJATSmetadataIfExists(element,["volume","issue","fpage","lpage","year"],newref)
        id=element.find("pub-id",{"pub-id-type":"doi"})
        if id:
            newref["doi"]=id.text
        id=element.find("pub-id",{"pub-id-type":"pmid"})
        if id:
            newref["pmid"]=id.text
        id=element.find("pub-id",{"pub-id-type":"pmc"})
        if id:
            newref["pmcid"]=id.text

        if newref["title"] == "":
            newref["title"]="<NO TITLE FOUND>"


##        comment=element.find("comment")
##        if comment:
##            extlink=comment.find("ext-link",{"ext-link-type":"uri"})
        extlinks=element.findAll("ext-link",{"ext-link-type":"uri"})
        for extlink in extlinks:
            newref["external_links"].append(extlink["xlink:href"])

        if original_id and self.USE_ORIGINAL_REF_ID:
            newref["id"]=original_id

        return newref

    def read(self, xml, identifier):
        """
            Load a JATS/NLM (PubMed) XML into a SciDoc.

            :param xml: full xml string
            :type xml: basestring
            :param identifier: an identifier for this document, e.g. file name
                        If an actual full path, the path will be removed from it
                        when stored
            :type identifier: basestring
            :returns: :class:`SciDoc <SciDoc>` object
            :rtype: SciDoc
        """
        # this solves a "bug" in BeautifulStoneSoup with "sec" tags
        BeautifulStoneSoup.NESTABLE_TAGS["sec"]=[]
        #xml=fixNumberCitationsXML(xml)
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
            newDocument["metadata"]["guid"]=cp.Corpus.generateGUID()
            return newDocument

        # Load metadata, either from corpus or from file
        self.loadJATSMetadataFromPaper(newDocument, soup)
        metadata["guid"]=cp.Corpus.generateGUID(metadata)

        # Load all references from the XML
        back=soup.find("back")
        if back:
            ref_list=back.find("ref-list")
            # other things in <back> like appendices: ignore them for now
            if ref_list:
                for ref in ref_list.findAll("ref"):
                    self.loadJATSReference(ref, newDocument)

        newDocument.updateReferences()

        # Load Abstract
        self.loadJATSAbstract(soup,newDocument)

        for sec in body.findChildren("sec", recursive=False):
            self.loadJATSSection(sec, newDocument, "root")

        newDocument.updateAuthorsAffiliations()
        return newDocument

def generateSideBySide(doc_list):
    """
        Generates side-by-side visualizations of a JATS XML: one using an XML to HTML
        converter, one loading the XML into SciDocJSON and rendering it back as HTML
    """
    from subprocess import Popen
    from read_auto import AutoXMLReader

    reader=AutoXMLReader()
    output_dir=os.path.join(cp.Corpus.ROOT_DIR,"conversion_visualization\\")

    file_list=[]
    for filename in doc_list:
        print("Converting %s" % filename)
        input_file=cp.Corpus.paths.inputXML+filename
        output_file=output_dir+"%s_1.html" % os.path.basename(filename)

##        os_line="..\\..\\libs\\generate_jats_html.bat "+" "+input_file+" "+output_file
##        print(os_line)
##        p = Popen(os_line, cwd=r"..\..\libs")
##        stdout, stderr = p.communicate()

        doc=reader.readFile(input_file)
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



def main():
    drive="g"
    cp.useLocalCorpus()
    cp.Corpus.connectCorpus(drive+":\\nlp\\phd\\pmc")

    debugging_files=[
    r"articles.I-N\J_Contemp_Brachytherapy\J_Contemp_Brachytherapy_2012_Sep_29_4(3)_176-181.nxml",
    r"articles.O-Z\PLoS_ONE\PLoS_One_2013_Dec_20_8(12)_e85076.nxml",
    r"articles.C-H\Gastroenterol_Rep_(Oxf)\Gastroenterol_Rep_(Oxf)_2013_Sep_17_1(2)_149-152.nxml",
    ]

##    debugging_files=cp.Corpus.selectRandomInputFiles(500,"*.nxml")
##    generateSideBySide(debugging_files)
##    inspectFiles(debugging_files)
    pass


if __name__ == '__main__':
    main()

