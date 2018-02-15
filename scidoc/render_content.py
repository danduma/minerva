# Functions to pretty-print SciDocJSON
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

# TODO use CSLRenderer
# TODO future: move all rendering to templates (jinja2)

from __future__ import absolute_import
from .reference_formatting import CSLRenderer, formatReference, formatCitation, formatAPACitationAuthors, formatAuthorNamePlain
import re
import six
from six.moves import range

metadata_labels={
    "journal_nlm-ta":"Journal NLM TA",
    "journal_iso-abbrev":"Journal ISO abbreviation",
    "publisher-id": "Publisher ID",
    "publisher-name": "Publisher name",
    "fpage": "First page",
    "lpage": "Last page",
    "volume": "Volume",
    "pmc_id": "PubMedCentral ID",
    "filename": "File name",
    "doi": "DOI",
    "title": "Title",
    "year": "Year",
}

class SciDocRenderer(object):
    """
        Renderer object. Encapsulates all the functions that deal with pretty
        printing the SciDoc.

        Note: a bit hackish, needs refactoring.
    """
    # TODO: make a class that can be subclassed to override formatting functions etc.
    def __init__(self, doc):
        """
        """
        self.doc=doc

    def prettyPrintDocumentMetadata(self):
        """
            Return a HTML representation of the metadata
        """

        res=[]
        for key in metadata_labels:
            if key in self.doc.metadata:
                res.append(u"<p>%s: %s</p>" % (metadata_labels[key], self.doc.metadata[key]))

        res.append("<h2>Authors</h2>")

        for author in self.doc.metadata["authors"]:
            res.append(u"<p>%s</p>" % formatAuthorNamePlain(author))
        return"".join(res)

    def prettyPrintSentenceHTML(self, s, text_formatting_function=lambda x:x, glob=None):
        """
            Fill in the <cit> placeholders with a nice representation of each reference, and
            call formatting function on each
        """
        text=s["text"]
        for count in range(2):
            text=re.sub(r"(<cit.*?/>)\s*?(<cit.*?/>)",r"\1, \2",text,0,re.IGNORECASE|re.DOTALL)

        text=re.sub(r"<cit\sid=\"?(.+?)\"?\s*?/>",r"__cit__\1",text, flags=re.DOTALL|re.IGNORECASE)
        text=text_formatting_function(text, glob)

        for cit_id in s.get("citations",[]):
            match=self.doc.matchReferenceByCitationId(cit_id)
            if match:
                sub=formatCitation(match)
##                sub=u"<span class=\"citation\" id=\"%s\" ref_id=\"%s\">%s</span>" % (cit_id, match["id"], sub)
                sub=u"<a href=\"#%s\" class=\"citation\" id=\"%s\">%s</a>" % (match["id"], cit_id, sub)
                sub=glob["citation_formatting_function"](sub, match)
            else:
                sub="[MISSING REFERENCE "+cit_id+")] "

            text=re.sub(r"__cit__"+str(cit_id),sub,text, flags=re.DOTALL|re.IGNORECASE)

        text=text.strip()
        if len(text) > 0 and text[-1] not in [".",",","!","?",":",";"]:
            text+=u"."
        return text

    def prettyPrintDocumentHTML( self,
        formatspans=False,                  # should the extra_attribute_function be called?
        include_bibliography=True,          # add the references section?
        include_metadata=False,             # show all metadata in the document at the beginning?
        wrap_with_HTML_tags=True,           # should the HTML code be wrapped inside <html> tags?
        text_formatting_function=None,      # function that processes the whole text of a sentence
        reference_formatting_function=None, # format the reference (at the end)
        citation_formatting_function=None,  # format the in-text citation
        extra_attribute_function=None       # adds attributes to the <span> tag
        ):
        """
            Pretty-prints a document held in a Scholarly JSON Document format (SJD)
        """

        def getAttributesString(element,attrs,glob):
            """
                Calls the extra attribute function and returns the string to be
                added to the XML tag with the attributes
            """
            glob["extra_attribute_function"](element,attrs,glob)
            res=""
            for key in attrs:
                res+=key+"='"+str(attrs[key])+"' "
            res=res.strip()
            return res

        def closeOpenTags(glob):
            """
                returns a list of concatenated closing tags for the ones open
                in glob["open_tags"]=[]
            """
            res=""
            open_tags=glob.get("open_tags",[])
            open_tags.reverse()

            for tag in open_tags:
                res+=u"</%s>" % tag

            glob["open_tags"]=[]
            return res

        def addHeader(section, embedding_level, glob):
            if glob["formatspans"]:
                glob["h"]+=1
                text=glob["text_formatting_function"](section["header"],glob)
##                return "<h"+str(embedding_level)+" id='head-"+str(section["id"])+"'>"+text+"</h"+str(embedding_level)+">"
                return u"<h%d id=\"head-%s\">%s</h%d>" % (embedding_level, section["id"], text, embedding_level)
            else:
##                return "<h"+str(embedding_level)+">"+section["header"]+u"</h"+str(embedding_level)+">"
                return u"<h%d>%s</h%d>" % (embedding_level, section["header"], embedding_level)

        def addSentenceHTML(s, glob):
            """
                Returns a formatted sentence
            """
            text=self.prettyPrintSentenceHTML(s, glob["text_formatting_function"], glob)
            if glob["formatspans"]:
                glob["s"]+=1
                attr_text=getAttributesString(s,{"id":"sent-"+str(s["id"])},glob)
                res=u"<span %s>%s</span>&nbsp;" % (attr_text,text)
            else:
                res=u"<span>%s</span>&nbsp;" % text
            return res

        def addParagraphHTML(p, glob):
            """
                Returns a formatted paragraph, dealing with those that are list items as well
            """
            LIST_TYPE="ptype"
            if glob["formatspans"]:
                glob["p"]+=1
                res_fmt=u"<p id=\""+six.text_type(p["id"])+u'">%s</p>'
            else:
                res_fmt=u"<p>%s</p"

            result=""
            open_tags=glob.get("open_tags",[])
            list_type=p.get(LIST_TYPE,"")
            if list_type in ["ul","ol"]:
                res_fmt=u'<li id="'+six.text_type(p["id"])+u'">%s</li>'
                if len(open_tags) > 0:
                    if open_tags[-1] != list_type:
                        res_fmt+="<"+list_type+">"+res_fmt
                        open_tags.append(list_type)
                        glob["open_tags"]=open_tags
                else:
                    glob["open_tags"].append(list_type)
            else:
                closeOpenTags(glob)


            for s in p["content"]:
                sent=self.doc.element_by_id[s]
                if sent["type"]=="s":
                    result+=addSentenceHTML(sent, glob)

            return res_fmt%result

        def processElement(element,embedding_level,glob):
            """
                Deals with sections and paragraphs, returns the full formatted
                HTML for them
            """
            result=[]
            if self.doc.isSection(element):
                result.append(closeOpenTags(glob))
                result.append(recurseSections(element, embedding_level, glob))
            if self.doc.isParagraph(element):
                result.append(addParagraphHTML(element,glob))
            return "".join(result)

        def recurseSections(section, embedding_level, glob):
            """
                Deals with a section
            """
            embedding_level+=1
            result=[]
            result.append(addHeader(section, embedding_level, glob))

            for element in section["content"]:
                result.append(processElement(self.doc.element_by_id[element],embedding_level,glob))

            result.append(closeOpenTags(glob))
            embedding_level-=1
            return u"".join(result)

        def recurseSectionsBibliography(biblos, embedding_level, glob):
            embedding_level+=1
            res=""

            if isinstance(biblos,dict) and "references" in biblos:
                if "header" in biblos:
                    res += addHeader (biblos, embedding_level, glob)
                thelist=biblos["references"]
            elif isinstance(biblos,list):
                thelist=biblos
            else:
                raise ValueError("Bibliography in the wrong format")

            for ref in thelist:
                if ref["type"]=="subsection":
                    res+=recurseSectionsBibliography(ref, embedding_level, glob)
                elif ref["type"]=="reference":
                    if ref.get("text","") != "":
                        reftext=ref["text"]
                    else:
                        reftext=formatReference(ref)
                    html_ref=glob["reference_formatting_function"](reftext,glob,ref)
                    res+=html_ref
            embedding_level-=1
            return res

        def defaultReferenceFormatting(text, glob, ref):
            """
                Default function.
            """
            return "<span class=\"reference\"><a name=\"%s\"></a>%s</span>" % (ref["id"],text)


        embedding_level=0
        result=[]
        result.append(u"<div id='document-text'>")
        result.append(u"<h1 class=\"title\">%s</h1>" % self.doc.data["metadata"]["title"])
        result.append(u"<div id=\"authors\">%s,%s</div>" % (formatAPACitationAuthors(self.doc.data["metadata"]),self.doc.data["metadata"]["year"]+""))
        glob={"doc":self.doc,"s":0,"p":0,"h":0,
        "formatspans":formatspans,
        "text_formatting_function":text_formatting_function if text_formatting_function else lambda x,y:x,
        "reference_formatting_function":reference_formatting_function if reference_formatting_function else defaultReferenceFormatting,
        "citation_formatting_function":citation_formatting_function if citation_formatting_function else lambda x,y:x,
        "extra_attribute_function":extra_attribute_function if extra_attribute_function else lambda x,y,z:y}

        if include_metadata:
            result.append(u"<h2>Metadata</h2>")
            result.append(self.prettyPrintDocumentMetadata())

        for element in self.doc.allsections:
            if self.doc.isSection(element) and element["parent"]=="root":
                result.append(recurseSections(element, embedding_level, glob))

        result.append(u"</div>")
        biblio_add=""

        if include_bibliography:
            embedding_level+=1
            biblio_add+=recurseSectionsBibliography(self.doc.data["references"], embedding_level, glob)

            if len(biblio_add) > 0:
                result.append(addHeader({"header":"Bibliography", "id":-999}, embedding_level, glob))
                result.append(biblio_add)
            embedding_level-=1

        result_text="".join(result)
        if wrap_with_HTML_tags:
            result_text=u"<html><head><meta charset=\"utf-8\" /> <link href=\"scidocview.css\" rel='stylesheet'></head><body>%s</body></html>" % result_text

        self.doc.glob=glob
        return result_text

def main():
    pass

if __name__ == '__main__':
    main()
