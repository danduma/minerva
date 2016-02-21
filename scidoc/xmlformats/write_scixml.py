#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      dd
#
# Created:     27/03/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from scidoc import *
import re, codecs

from general_utils import safe_unicode
from base_classes import BaseSciDocXMLreader,BaseSciDocXMLwriter

class SciXMLWriter(BaseSciDocXMLwriter):
    def __init__(self):
        """
        """
        self.header_depth=0

    def dumpMetadata(self, doc):
        """
        """
        lines=[]
        lines.append("<METADATA>")
        if doc["metadata"].has_key("fileno"):
            lines.append("<FILENO>"+doc.metadata["fileno"]+"</FILENO>\n")

        lines.append("<FILENAME>"+doc.metadata["filename"]+"</FILENAME>\n")

        lines.append("<APPEARED>")
        if "conference" in doc.metadata:
            lines.append("<CONFERENCE>"+doc.metadata["conference"]+"</CONFERENCE>\n")
        if "journal" in doc.metadata:
            lines.append("<JOURNAL>"+doc.metadata["journal"]+"</JOURNAL>\n")

        lines.append(u"<CURRENT_AUTHORLIST>")
        for author in doc.metadata["authors"]:
            bits=author.split(",")
            rest=" ".join(bits[1:])
            lines.append(u"<CURRENT_AUTHOR>")
##            lines.append("<FIRSTNAME>%s</FIRSTNAME>" % rest)
            lines.append(rest+u"<CURRENT_SURNAME>%s</CURRENT_SURNAME>" % bits[0])
            lines.append(u"</CURRENT_AUTHOR>")
        lines.append(u"</CURRENT_AUTHORLIST>")

##        lines.append("<SURNAMES>")
##        for author in doc["authors"]:
##
##        lines.append("</SURNAMES>")

        lines.append(u"<YEAR>"+doc.metadata["year"]+"</YEAR>")
        lines.append(u"</APPEARED>")

        if doc["metadata"].has_key("revisionhistory"):
            lines.append(u"<REVISIONHISTORY>"+doc.metadata["revisionhistory"]+u"</REVISIONHISTORY>\n")
        lines.append(u"</METADATA>")
        return lines

    def dumpAbstract(self, doc):
        """
        """
        lines=[]
        lines.append(u"<ABSTRACT>")
        if "content" in doc.abstract:
            for element_id in doc.abstract["content"]:
                s=doc.element_by_id[element_id]
                if s["type"]=="s":
                    lines.append(self.dumpSentence(doc,s, True))
                if s["type"]=="p":
                    lines.extend(self.dumpParagraph(doc,s, True))
        lines.append(u"</ABSTRACT>")
        return lines


    def dumpSentence(self, doc,s,in_abstract=False):
        """
            Fixes all the contents of the sentence, returns a string of proper XML
        """

        def repl_func(m):
            """
                Replaces <CIT ID=0 /> with <REF ID=cit0 REFID=ref0 />
            """
            cit_id=m.group(1) # this is the actual unique identifier of a citation in the document
            ref_id=doc.citation_by_id[cit_id]["ref_id"] # this is the id of the reference the cit cites
            return u'<REF ID="'+safe_unicode(cit_id)+u'" REFID="'+safe_unicode(ref_id)+'" />'

        if in_abstract:
            xmltag=u"A-S"
        else:
            xmltag=u"S"

        res=u"<"+xmltag+u' ID="%s"' % s["id"]

        if "az" in s:
            res+=u'AZ="%s"' % s["az"]

        text=s["text"]
        text=re.sub(r"<CIT ID=(.*?)\s?/>",repl_func, text)
        res+=">"+text+"</"+xmltag+">"

        return res

    def dumpParagraph(self, doc,paragraph, in_abstract=False):
        """
        """
        lines=[]

        # fix for empty paragraphs: should it be here?
        if len(paragraph["content"]) == 0:
            return lines

        if not in_abstract:
            lines.append(u"<P>")
        for element in paragraph["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="s":
                lines.append(self.dumpSentence(doc,element, in_abstract))

        if not in_abstract:
            lines.append(u"</P>")
        return lines

    def dumpSection(self, doc,section, header_depth):
        """
        """
        self.header_depth+=1
        lines=[]
        lines.append(u'<DIV depth="'+str(header_depth)+'">')
        lines.append(u"<HEADER>%s</HEADER>" % section["header"])
        for element in section["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="section":
                lines.extend(self.dumpSection(doc,element, header_depth))
            elif element["type"]=="p":
                lines.extend(self.dumpParagraph(doc,element))

        lines.append(u"</DIV>")
        self.header_depth-=1
        return lines

    def dumpBody(self, doc):
        """
        """
        lines=[]
        lines.append(u"<BODY>")
        for section in doc.allsections:
            if section != doc.abstract:
                lines.extend(self.dumpSection(doc,section,0))
        lines.append(u"</BODY>")
        return lines

    def dumpRefAuthor(self, doc,author):
        """
        """
        lines=[]
        lines.append(u"<AUTHOR>")
        if isinstance(author,basestring):
            lines.append(author)
        elif isinstance(author,dict):
            lines.append(author["given"])
            lines.append(u"<SURNAME>%s</SURNAME>" % author["family"])
        lines.append(u"</AUTHOR>")
        return lines

    def dumpRefAuthors(self, doc, ref):
        """
        """
        lines=[]
        lines.append("<AUTHORLIST>")
        for author in ref["authors"]:
            lines.extend(self.dumpRefAuthor(doc,author))
        lines.append("</AUTHORLIST>")
        return lines

    def dumpReference(self, doc, ref):
        """
        <REFERENCE ID="cit1">
        <AUTHORLIST>
        <AUTHOR>L.<SURNAME>Que, Jr.</SURNAME></AUTHOR>
        <AUTHOR>R. Y. N.<SURNAME>Ho</SURNAME></AUTHOR>
        </AUTHORLIST>
        <TITLE></TITLE>
        <JOURNAL>Chem. Rev.<YEAR>1996</YEAR>962607-2624</JOURNAL></REFERENCE>
        """
        lines=[]
        lines.append('<REFERENCE ID="%s">' % ref["id"])
        lines.extend(self.dumpRefAuthors(doc,ref))
        lines.append("<TITLE>%s</TITLE>" % ref["title"])
        lines.append("<YEAR>%s</YEAR>" % ref["year"])
        if "journal" in ref:
            lines.append("<JOURNAL>%s</JOURNAL>" % ref["journal"])
        lines.append("</REFERENCE>")
        return lines

    def dumpReferences(self, doc):
        """
        """
        lines=[]
        lines.append("<REFERENCELIST>")
        for ref in doc.data["references"]:
            lines.extend(self.dumpReference(doc, ref))
        lines.append("</REFERENCELIST>")
        return lines

    def write(self, doc, filename):
        """
        """
        lines=[]
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<!DOCTYPE PAPER SYSTEM "paper-structure-annotation.dtd">')

        lines.append("<PAPER>")
        lines.extend(self.dumpMetadata(doc))
        lines.extend(self.dumpAbstract(doc))
        lines.extend(self.dumpBody(doc))
        lines.extend(self.dumpReferences(doc))
        lines.append("</PAPER>")

        lines2=[]
        for line in lines:
            lines2.append(safe_unicode(line))

        lines=lines2
    ##    text="\n".join(lines)
        f=codecs.open(filename,"w", encoding="utf-8",errors="ignore")
    ##    f.writelines([line+"\n" for line in lines])
        f.writelines(lines)
        f.close()
    ##    return text

def saveSciXML(doc,filename):
    """
        Wrapper around SciXMLWriter to preserve backwards compatibility
    """
    writer=SciXMLWriter()
    writer.write(doc, filename)

def main():
    from azscixml import loadAZSciXML
    from scidoc import SciDoc
    doc=SciDoc(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
##    doc=loadAZSciXML(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
    saveSciXML(doc,r"C:\NLP\PhD\bob\output\\"+doc["metadata"]["filename"]+".xml")
    pass

if __name__ == '__main__':
    main()

def saveSciXML(doc,filename):
    """
        Exports a SciDocJSON to a file in SciXML format
    """
