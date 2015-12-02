# Basic reference formatting functions.
#
# Copyright (C) 2015 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import (absolute_import, division, print_function,
##                        unicode_literals
                        )
from citeproc.py2compat import *

# We'll use json.loads for parsing the JSON data.
import json, os, re

# Import the citeproc-py classes we'll use below.
from citeproc import CitationStylesStyle, CitationStylesBibliography
from citeproc import Citation, CitationItem
from citeproc import formatter
from citeproc.source.json import CiteProcJSON

CSL_PATH = re.sub(r"scidoc.*.py",r"cit_styles",__file__,flags=re.IGNORECASE)+os.sep

class CSLRenderer(object):
    """
        Interface with citeproc-py CSL library
    """
    def __init__(self, doc, style):
        self.citations={}
        self.doc=doc
        style=style.lower()

        bib_source = self.convertReferencesToJSON()
        bib_source = CiteProcJSON(bib_source)
        bib_style = CitationStylesStyle(os.path.join(CSL_PATH,style), validate=False)

        self.bibliography = CitationStylesBibliography(bib_style, bib_source, formatter.html)
        self.prepareCitations()

    def convertReferencesToJSON(self):
        """
            Converts a SciDoc's references into the JSON format expected
        """
        def copyFields(dict1, dict2, field_list):
            for field in field_list:
                if field[0] in dict1:
                    dict2[field[1]]=dict1[field[0]]

        res=[]
        for ref in self.doc.references:
            newref={}
            copyFields(ref, newref,
                # new = old
                [("id","id"),
                ("title","title"),
                ("authors","author"),
                ("publisher","publisher-name"),
                ("title","title"),
                ("type","type"),
                ("URL","url"),
                ])
            newref["issued"]={"date-parts":[(ref.get("year","0"),)]}
            res.append(newref)
        return res

    def prepareCitations(self):
        """
        """
        for cit in self.doc.citations:
            self.citations[cit["id"]]=Citation([CitationItem(cit["ref_id"])])
            self.bibliography.register(self.citations[cit["id"]])

    def getCitationText(self, cit):
        """
        """
        warn=lambda x:None
        return self.bibliography.cite(self.citations[cit["id"]], warn)

    def getBibliography(self):
        """
            Returns the formatted bibliography
        """
        return [str(item) for item in self.bibliography.bibliography()]


# -----------------------------------------------------------------------------
#  Basic, hackish reference formatting functions. Recommended to use CSLRenderer
#  and predefined .CSL files
# -----------------------------------------------------------------------------

def getAuthorReferenceNameAPA(author):
    """
        Returns J. Smith

        >>> getAuthorReferenceNameAPA({"given":"John","family":"Smith"})
        J. Smith
    """
    assert(isinstance(author,dict))
    names=author.get("given","").split()
    initials=". ".join([name[0] for name in names]).strip()+"."
    return "%s %s" % (initials, author["family"])

def formatListOfAuthors(authors, style="APA"):
    """
        This is meant to return a list of authors for reference formatting.
    """
    if style=="APA":
        result=""

        if authors == []:
            result+=u"?"
        elif len(authors) == 1:
            result += u'%s ' % getAuthorReferenceNameAPA(authors[0])
        elif len(authors) == 2:
            result += u'%s and %s' % (getAuthorReferenceNameAPA(authors[0]), getAuthorReferenceNameAPA(authors[1]))
        else:
            for index,author in enumerate(authors):
                if index == len(authors)-2:
                    result += getAuthorReferenceNameAPA(author) + " and "
                else:
                    result += getAuthorReferenceNameAPA(author) + ", "
        return result.strip().strip(",")
    return None


def formatReference(ref, style="APA"):
    """
        Formats a reference/metadata in plain text (for references section)
    """
    if style=="APA":
        authors=formatListOfAuthors(ref.get("authors",[]))
        apa_line= "%s. %s. %s." % \
            (authors, str(ref.get("year","")),
             ref.get("title",""))
        if ref.get("publication","") != "":
            apa_line+=" %s." % ref["publication"]
        if ref.get("volume","") != "":
            apa_line+=" %s." % ref["volume"]
        if ref.get("pages","") != "":
            apa_line+=" %s." % ref["pages"]
        if ref.get("location","") != "":
            apa_line+=" %s." % ref["location"]
        if ref.get("publisher","") != "":
            apa_line+=" %s." % ref["publisher"]
        return apa_line
    return None

def formatCitation(refValues, style="APA"):
    """
        Format an in-text reference

        Can deal with either a detailed dict of authors or just a list of surnames
    """
    authors=refValues.get('surnames',None)
    if not authors:
        authors=refValues.get('authors',[])
    res=u""
    if authors == []:
        res+=u"?"
    elif len(authors) == 1:
        if isinstance(authors[0], dict):
            res += u'%s ' % authors[0]["family"]
        elif isinstance(authors[0], basestring):
            res += u'%s ' % authors[0]
    elif len(authors) == 2:
        if isinstance(authors[0], dict):
            res += u'%s and %s' % (authors[0]["family"], authors[1]["family"])
        elif isinstance(authors[0], basestring):
            res += u'%s and %s' % (authors[0], authors[1])
    else:
        if isinstance(authors[0], dict):
            res += u'%s et al.' % authors[0]["family"]
        elif isinstance(authors[0], unicode):
            res += u'%s et al.' % authors[0]
        elif isinstance(authors[0], str):
            res += u'%s et al.' % unicode(authors[0], errors="replace")

    res+=" (%s)" % refValues.get("year", "?")
    res=res.replace("  "," ")
    return res


def formatAuthorNamePlain(author):
    """
        Given an author dict, returns a simple string with their name
    """
    return "%s %s" % (author.get("given",""),author.get("family",""))

def formatAPACitationAuthors(refValues):
    """
        Return only the authors for a citation formatted for APA-style bibliography

        VERY HACKISH, using only surnames
    """
    authors=refValues.get('surnames',[])
    if authors==[]: authors=refValues.get('authors',[])

    res=u""
    if authors == []:
        res+=u"?"
    elif len(authors) == 1:
        res += u'%s ' % authors[0]
    elif len(authors) == 2:
        res += u'%s and %s' % (authors[0], authors[1])
    else:
        try:
            res += authors[0] + u' et al.'
        except:
            res += unicode(authors[0],errors="replace")
    return res

def formatAPACitation(refValues):
    """
        Return a citation formatted for APA-style bibliography
    """
    res=formatAPAReferenceAuthors(refValues)

    res+=u" " + unicode(refValues["year"])

    try:
        res+=u". "+ refValues["title"]
    except:
        res+=u". "+ unicode(refValues["title"],errors="replace")
    return res


def basicTest():
    print (__file__)
    import minerva.db.corpora as corpora
    drive="g"
    corpora.Corpus.connectCorpus(drive+":\\nlp\\phd\\pmc")

    from minerva.util.general_utils import loadFileText
    from minerva.xmlformats.read_jatsxml import JATSXMLReader
    reader = JATSXMLReader()
    doc=reader.read(loadFileText(r"G:\NLP\PhD\pmc\inputXML\articles.O-Z\PLoS_ONE\\PLoS_One_2013_Dec_20_8(12)_e85076.nxml"),"one")

##    renderer=CSLRenderer(doc,".." + os.sep + "cit_styles" + os.sep + 'ama.csl')
    renderer=CSLRenderer(doc,"ama")

    print("Citations\n\n")
    for cit in doc.citations:
        print(renderer.getCitationText(cit))

    print("Bibliography\n\n")
    for line in renderer.getBibliography():
        print(line)


def main():
    basicTest()
    pass

if __name__ == '__main__':
    main()
