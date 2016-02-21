#-------------------------------------------------------------------------------
# Name:        export_jats
# Purpose:
#
# Author:      dd
#
# Created:     27/03/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

"""
    WARNING WARNING WARNING

    Not fully implemented, mostly from export_scixml.py !
    implemented metadata & abstract so far
"""


from scidoc import *
import re

def saveJATS_XML(doc,filename):
    """
        Exports a SciDocJSON to a file in JATS XML format
    """
    header_depth=0

    def dumpMetadata(doc):
        """
        """
        lines=[]
        lines.append("<METADATA>")
        if doc["metadata"].has_key("fileno"):
            lines.append("<FILENO>"+doc["metadata"]["fileno"]+"</FILENO>\n")

        lines.append("<FILENAME>"+doc["filename"]+"</FILENAME>\n")


        lines.append("<APPEARED>")
        if "conference" in doc.metadata:
            lines.append("<CONFERENCE>"+doc["metadata"]["conference"]+"</CONFERENCE>\n")
        if doc["metadata"].has_key("journal"):
            lines.append("<JOURNAL>"+doc["metadata"]["journal"]+"</JOURNAL>\n")

        lines.append("<AUTHORS>")
        for author in doc["authors"]:
            bits=author.split(",")
            rest=" ".join(bits[1:])
            lines.append("<AUTHOR>")
            lines.append("<FIRSTNAME>%s</FIRSTNAME>" % rest)
            lines.append("<SURNAME>%s</SURNAME>" % bits[0])
            lines.append("</AUTHOR>")
        lines.append("</AUTHORS>")

##        res+="<SURNAMES>"
##        for author in doc["authors"]:
##
##        res+="</SURNAMES>"

        lines.append("<YEAR>"+doc["year"]+"</YEAR>")
        lines.append("</APPEARED>")

        if doc.metadata.has_key("revisionhistory"):
            lines.append("<REVISIONHISTORY>"+doc["metadata"]["revisionhistory"]+"</REVISIONHISTORY>\n")

        return lines


    def dumpAbstract(doc):
        """
        """
        lines=[]
        lines.append("<ABSTRACT>")
        for element_id in doc.abstract["content"]:
            s=doc.element_by_id[element_id]
            if s["type"]=="s":
                lines.append(dumpSentence(doc,s, True))
            if s["type"]=="p":
                lines.extend(dumpParagraph(doc,s, True))
        lines.append("</ABSTRACT>")
        return lines


    def dumpSentence(doc,s,in_abstract=False):
        """
            Fixes all the contents of the sentence, returns a string of proper XML
        """

        def repl_func(m):
            """
                Replaces <CIT ID=0 /> with <REF ID=cit0 REFID=ref0 />
            """
            cit_id=m.group(1) # this is the actual unique identifier of a citation in the document
            ref_id=doc.citation_by_id[cit_id]["ref_id"] # this is the id of the reference the cit cites
            return '<REF ID="'+cit_id+'" REFID="'+ref_id+'" />'

        if in_abstract:
            xmltag="A-S"
        else:
            xmltag="S"

        res="<"+xmltag+' ID="%s"' % s["id"]

        if "az" in s:
            res+='AZ="%s"' % s["az"]

        text=s["text"]
        text=re.sub(r"<CIT ID=(.*?)\s?/>",repl_func, text)
        res+=">"+text+"</"+xmltag+">"

        return res

    def dumpParagraph(doc,paragraph, in_abstract=False):
        """
        """
        lines=[]

        # fix for empty paragraphs: should it be here?
        if len(paragraph["content"]) == 0:
            return lines

        if not in_abstract:
            lines.append("<P>")
        for element in paragraph["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="s":
                lines.append(dumpSentence(doc,element, in_abstract))

        if not in_abstract:
            lines.append("</P>")
        return lines

    def dumpSection(doc,section, header_depth):
        """
        """
        header_depth+=1
        lines=[]
        lines.append('<DIV depth="'+str(header_depth)+'">')
        lines.append("<HEADER>%s</HEADER>" % section["header"])
        for element in section["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="section":
                lines.extend(dumpSection(doc,element, header_depth))
            elif element["type"]=="p":
                lines.extend(dumpParagraph(doc,element))

        lines.append("</DIV>")
        header_depth-=1
        return lines

    def dumpBody(doc):
        """
        """
        lines=[]
        lines.append("<BODY>")
        for section in doc.allsections:
            if section != doc.abstract:
                lines.extend(dumpSection(doc,section,0))
        lines.append("</BODY>")
        return lines

    def dumpRefAuthor(doc,author):
        """
        """
        lines=[]
        lines.append("<AUTHOR>")
        if isinstance(author,basestring):
            lines.append(author)
        elif isinstance(author,dict):
            lines.append(author["given"])
            lines.append("<SURNAME>%s</SURNAME>" % author["family"])
        lines.append("</AUTHOR>")
        return lines

    def dumpRefAuthors(doc,ref):
        """
        """
        lines=[]
        lines.append("<AUTHORLIST>")
        for author in ref["authors"]:
            lines.extend(dumpRefAuthor(doc,author))
        lines.append("</AUTHORLIST>")
        return lines

    def dumpReference(doc,ref):
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
        lines.extend(dumpRefAuthors(doc,ref))
        lines.append("<TITLE>%s</TITLE>" % ref["title"])
        lines.append("<YEAR>%s</YEAR>" % ref["year"])
        if "journal" in ref:
            lines.append("<JOURNAL>%s</JOURNAL>" % ref["journal"])
        lines.append("</REFERENCE>")
        return lines

    def dumpReferences(doc):
        """
        """
        lines=[]
        lines.append("<REFERENCELIST>")
        for ref in doc.data["references"]:
            lines.extend(dumpReference(doc,ref))
        lines.append("</REFERENCELIST>")
        return lines

    lines=[]

    # !TODO change all of this to JATS
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE PAPER SYSTEM "paper-structure-annotation.dtd">')

    lines.append("<PAPER>")
    lines.extend(dumpMetadata(doc))
    lines.extend(dumpAbstract(doc))
    lines.extend(dumpBody(doc))
    lines.extend(dumpReferences(doc))
    lines.append("</PAPER>")

##    text="\n".join(lines)
    f=open(filename,"w")
    f.writelines([line+"\n" for line in lines])
    f.close()
##    return text

def saveJATS_XML(doc, filename):
    """
    """

    def dumpAbstract(doc):
        """
        """
        res="<ABSTRACT>"
        for s in doc["abstract"]["sentences"]:
            res+="<A-S ID='%s'>" % s["id"]
        res+="</ABSTRACT>"


    s="<DOC>\n<TEXT>"
    s+=u"<TITLE>%s</TITLE>" % doc["title"]



def main():
    from azscixml import loadAZSciXML
    from scidoc import SciDoc
    doc=SciDoc(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
##    doc=loadAZSciXML(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
    saveSciXML(doc,r"C:\NLP\PhD\bob\output\\"+doc["metadata"]["filename"]+".xml")
    pass

if __name__ == '__main__':
    main()
