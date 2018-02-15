#-------------------------------------------------------------------------------
# Name:        export_sapientaxml
# Purpose:      export the particular flavour of SciXML that SAPIENTA takes as input
#
# Author:      dd
#
# Created:     27/03/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from scidoc import *
import re, codecs

from general_utils import safe_unicode
from cStringIO import StringIO
import six

html_escape=["quot", "amp", "apos", "lt", "gt", "nbsp", "iexcl", "cent", "pound", "curren", "yen", "brvbar", "sect", "uml", "copy", "ordf", "laquo", "not", "shy", "reg", "macr", "deg", "plusmn", "sup2", "sup3", "acute", "micro", "para", "middot", "cedil", "sup1", "ordm", "raquo", "frac14", "frac12", "frac34", "iquest", "Agrave", "Aacute", "Acirc", "Atilde", "Auml", "Aring", "AElig", "Ccedil", "Egrave", "Eacute", "Ecirc", "Euml", "Igrave", "Iacute", "Icirc", "Iuml", "ETH", "Ntilde", "Ograve", "Oacute", "Ocirc", "Otilde", "Ouml", "times", "Oslash", "Ugrave", "Uacute", "Ucirc", "Uuml", "Yacute", "THORN", "szlig", "agrave", "aacute", "acirc", "atilde", "auml", "aring", "aelig", "ccedil", "egrave", "eacute", "ecirc", "euml", "igrave", "iacute", "icirc", "iuml", "eth", "ntilde", "ograve", "oacute", "ocirc", "otilde", "ouml", "divide", "oslash", "ugrave", "uacute", "ucirc", "uuml", "yacute", "thorn", "yuml", "OElig", "oelig", "Scaron", "scaron", "Yuml", "fnof", "circ", "tilde", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigmaf", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "thetasym", "upsih", "piv", "ensp", "emsp", "thinsp", "zwnj", "zwj", "lrm", "rlm", "ndash", "mdash", "lsquo", "rsquo", "sbquo", "ldquo", "rdquo", "bdquo", "dagger", "Dagger", "bull", "hellip", "permil", "prime", "Prime", "lsaquo", "rsaquo", "oline", "frasl", "euro", "image", "weierp", "real", "trade", "alefsym", "larr", "uarr", "rarr", "darr", "harr", "crarr", "lArr", "uArr", "rArr", "dArr", "hArr", "forall", "part", "exist", "empty", "nabla", "isin", "notin", "ni", "prod", "sum", "minus", "lowast", "radic", "prop", "infin", "ang", "and", "or", "cap", "cup", "int", "there4", "sim", "cong", "asymp", "ne", "equiv", "le", "ge", "sub", "sup", "nsub", "sube", "supe", "oplus", "otimes", "perp", "sdot", "lceil", "rceil", "lfloor", "rfloor", "lang", "rang", "loz", "spades", "clubs", "hearts", "diams"]
rx_html_escape="&("+"|".join(html_escape).strip("|")+")[^;]"
print(rx_html_escape)


def fixAmpersandsHTML(text):
    """
        Substitutes known &amp; HTML escapes missing their semicolon with the
        version with semicolon in, replaces all other &text with &amp;text
    """
##    text=re.sub(rx_html_escape,r"&\1;",text)
    text=re.sub("&([\w]{2,8})[^;\w]",r"&amp;\1",text)
    return text

def saveSapientaXML(doc,filename):
    """
        Exports a SciDocJSON to a file in Sapienta sort-of SciXML format
    """
    header_depth=0

    def dumpMetadata(doc):
        """
        """
        lines=[]
        lines.append('<mode2 hasDoc="yes" name="'+doc.metadata["filename"].replace(".xml","")+'" version="597"/>')
        lines.append("<METADATA>")
        if "fileno" in doc["metadata"]:
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
            rest=fixAmpersandsHTML(rest+u" ")
            bits[0]=fixAmpersandsHTML(bits[0]+u" ").strip()
            lines.append(rest+u"<CURRENT_SURNAME>%s</CURRENT_SURNAME>" % bits[0])
            lines.append(u"</CURRENT_AUTHOR>")
        lines.append(u"</CURRENT_AUTHORLIST>")

##        lines.append("<SURNAMES>")
##        for author in doc["authors"]:
##
##        lines.append("</SURNAMES>")

        lines.append(u"<YEAR>"+doc.metadata["year"]+"</YEAR>")
        lines.append(u"</APPEARED>")

        if "revisionhistory" in doc["metadata"]:
            lines.append(u"<REVISIONHISTORY>"+doc.metadata["revisionhistory"]+u"</REVISIONHISTORY>\n")
        lines.append(u"</METADATA>")
        return lines

    def dumpAbstract(doc):
        """
        """
        lines=[]
        lines.append(u"<ABSTRACT>")
        if "content" in doc.abstract:
            for element_id in doc.abstract["content"]:
                s=doc.element_by_id[element_id]
                if s["type"]=="s":
                    lines.append(dumpSentence(doc,s, True))
                if s["type"]=="p":
                    lines.extend(dumpParagraph(doc,s, True))
        lines.append(u"</ABSTRACT>")
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
            return u'<REF ID="'+safe_unicode(cit_id)+u'" REFID="'+safe_unicode(ref_id)+'" />'

        in_abstract=False # hack because of sapienta difference vs SciXML
        if in_abstract:
            xmltag=u"A-S"
        else:
            xmltag=u"s"

        s_id=re.match(r"\w+?(\d+)",s["id"],re.IGNORECASE)
        assert(s_id)
        res=u"<"+xmltag+u' sid="%s"' % str(int(s_id.group(1)))

        # !TODO change this to export more markers, such as CFC and whatever
        if "az" in s:
            res+=u'AZ="%s"' % s["az"]

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
            lines.append(u"<P>")
        for element in paragraph["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="s":
                lines.append(dumpSentence(doc,element, in_abstract))

        if not in_abstract:
            lines.append(u"</P>")
        return lines

    def dumpSection(doc,section, header_depth):
        """
        """
        header_depth+=1
        lines=[]
        lines.append(u'<DIV depth="'+str(header_depth)+'">')
        lines.append(u"<HEADER>%s</HEADER>" % section["header"])
        for element in section["content"]:
            element=doc.element_by_id[element]
            if element["type"]=="section":
                lines.extend(dumpSection(doc,element, header_depth))
            elif element["type"]=="p":
                lines.extend(dumpParagraph(doc,element))

        lines.append(u"</DIV>")
        header_depth-=1
        return lines

    def dumpBody(doc):
        """
        """
        lines=[]
        lines.append(u"<BODY>")
        for section in doc.allsections:
            if section != doc.abstract:
                lines.extend(dumpSection(doc,section,0))
        lines.append(u"</BODY>")
        return lines

    def dumpRefAuthor(doc,author):
        """
        """
        lines=[]
        lines.append(u"<AUTHOR>")
        if isinstance(author,six.string_types):
            lines.append(author)
        elif isinstance(author,dict):
            lines.append(author["given"])
            lines.append(u"<SURNAME>%s</SURNAME>" % author["family"])
        lines.append(u"</AUTHOR>")
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


    lines.append(u'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
##    lines.append('<!DOCTYPE PAPER SYSTEM "paper-structure-annotation.dtd">')
    lines.append(u"<PAPER>")
    lines.extend(dumpMetadata(doc))
    lines.extend(dumpAbstract(doc))
    lines.extend(dumpBody(doc))
    lines.extend(dumpReferences(doc))
    lines.append(u"</PAPER>")

    lines2=[]
    for line in lines:
        lines2.append(safe_unicode(line))
    lines=lines2

##    file_str = StringIO()
##    file_str.writelines(lines)
##    full_str=file_str.getvalue()
    full_str=u"".join(lines)
##    full_str=fixAmpersandsHTML(full_str)

    f=codecs.open(filename,"w", encoding="utf-8",errors="ignore")
    f.write(full_str)
    f.close()


def main():
    from .azscixml import loadAZSciXML
    from scidoc import SciDoc
    doc=SciDoc(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
##    doc=loadAZSciXML(r"C:\NLP\PhD\bob\fileDB\jsonDocs\a00-1001.json")
    saveSapientaXML(doc,r"C:\NLP\PhD\bob\output\\"+doc["metadata"]["filename"]+".xml")
    pass

if __name__ == '__main__':
    main()
