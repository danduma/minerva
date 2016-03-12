# SciXML converting to SciDocJSON. This library assumes clean SciXML, not the
# Ritchie 2006 conversion. For a crazy attempt to recover data from noise, see bobscixml.py
#
# Copyright:   (c) Daniel Duma 2013
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import os, glob, re, codecs, json
import cPickle, random
from BeautifulSoup import BeautifulStoneSoup

from minerva.proc.general_utils import *

import minerva.db.scidoc as scidoc
import minerva.db.corpora as cp
from scidoc.xmlformats.base_classes import BaseSciDocXMLReader

all_azs=set([u'OTH', u'BKG', u'BAS', u'CTR', u'AIM', u'OWN', u'TXT']) # argumentative zones
all_ais=set([u'OTH', u'BKG', u'OWN']) # intellectual attribution

rxauthors=re.compile(r"(<REFERENCE>.*?\n)(.*?)(<DATE>)", re.IGNORECASE | re.DOTALL)
rxtitle=re.compile(r"(</DATE>.*?\n)(.*?)(\.|</REFERENCE>)", re.IGNORECASE | re.DOTALL)
rxdate=re.compile(r"(<DATE>)(.*?)(</DATE>)", re.IGNORECASE | re.DOTALL)

rxsingleauthor=re.compile(r"(<SURNAME>)(.*?)(</SURNAME>)", re.IGNORECASE | re.DOTALL)
##rxsingleyear=re.compile(r"\d{4}\w{0,1}", re.IGNORECASE | re.DOTALL)
rxsingleyear=re.compile(r"in\spress|to\sappear|forthcoming|submitted|\d{4}\w{0,1}", re.IGNORECASE | re.DOTALL)
rxwtwoauthors=re.compile(r"(\w+)\sand\s(\w+)", re.IGNORECASE | re.DOTALL)
rxetal=re.compile(r"(\w+)\set\sal", re.IGNORECASE | re.DOTALL)

##DOCS_TO_IGNORE=[
##"a83-1002.xml", # authorlist missing, tag doesn't close
##"a83-1023.xml", # title is a whole paragraph
##]
##DOC_LIST=[]


# leftovers from AZ loading, maybe useful?
azs=[]
ias=[]


def debugAddMessage(doc,prop,msg):
    """
        Prints a message and adds it to the specified tag of a document
    """
    print msg   # uncomment
    doc[prop]=doc.data.get(prop,"")+msg

# ------------------------------------------------------------------------------
#   Helper functions
# ------------------------------------------------------------------------------

def cleanxml(xmlstr):
    """
        Removes all XML/HTML tags
    """
    xmlstr=re.sub(r"</?.+?>"," ",xmlstr)
    xmlstr=xmlstr.replace("  "," ").strip()
    return xmlstr

def tryToExtractTitle(reftext):
    """
        If no <title> tag is available, try to strip everything else and make the remainder the title
    """
    reftext=re.sub(r"<reference.+?>","",reftext,0, re.IGNORECASE| re.DOTALL)
    reftext=re.sub(r"</reference>","",reftext,0, re.IGNORECASE| re.DOTALL)
    reftext=re.sub(r"<author>.*</author>","",reftext,0, re.IGNORECASE| re.DOTALL)
    reftext=re.sub(r"<(\w+).*?>.*?</\1>","",reftext,0, re.IGNORECASE| re.DOTALL)
    reftext=reftext.replace(", , , , and . . ","").replace(", , , and . .","")
    reftext=reftext.replace(" and . . ","").replace(" and  ().","").replace(" and  .","").replace(" and , ","")
    reftext=reftext.strip()
    return reftext

def processPlainTextAuthor(author):
    """
        Returns a dictionary with a processed author's name
    """

##    print author

    bits=author.split()
    res={"family":"","given":"", "text":author}

##    if "<surname>" in author.lower():
##        match=rxsingleauthor.search(author)
##        if match:
##            surname=match.group(2)

    if len(bits) > 1:
        res["family"]=bits[-1]
        res["given"]=bits[0]
        if len(bits) > 2:
            res["middlename"]=" ".join(bits[1:-2])
    elif len(bits) == 1:
        res["family"]=bits[0]
    else:
        pass

    return res

class SciXMLReader(BaseSciDocXMLReader):
    """
    """

    def processReference(ref, doc):
        """
            Process reference format, try to recover title, authors, date
        """

        lines=ref.__repr__()

        lines=lines.replace("<reference> ","<reference>\n")
        match=rxauthors.search(lines)
        authors=match.group(2).replace("\n","") if match else ""

    ##    surnames=set([x[1] for x in rxsingleauthor.findAll(lines)])
        surnames=[x[1] for x in rxsingleauthor.findall(lines)]

        match=rxdate.search(lines)
        date=match.group(2).replace("\n","") if match else ""

        match=rxtitle.search(lines)
        title=match.group(2).replace("\n","") if match else lines

        if authors=="" and len(surnames) > 0: authors=" ".join(list(surnames))

        newref={"text":lines, "authors":authors, "surnames":surnames, "title":title, "year": date}
        doc["references"].append(newref)

    def processReferenceXML(ref,doc, firstcall=True):
        """
            Load a reference from the bibliography section
        """

        def fuseReferences(doc, ref):
            """
            """
            prevref=doc["references"][-1]
            doc["metadata"]["ref_replace_list"]=doc["metadata"].get("ref_replace_list",{})
            id=""
            try:
                id=ref["id"]
                if not id:
                    id=prevref["id"]
                    if isinstance(id,basestring):
                        id="ref"+str(len(doc["references"])+1)
                    elif isinstance(id,int):
                        id=id+1
            except:
                id="ref"+str(len(doc["references"])+1)

            doc["metadata"]["ref_replace_list"][id]=prevref["id"]
            doc["references"].remove(prevref)

            fullstring=re.sub(r"</reference>","",prevref["xml"],0,re.IGNORECASE)
            fullstring+=re.sub(r"<reference.+?>","",ref.__repr__(),0,re.IGNORECASE)
    ##                ref=BeautifulStoneSoup(prevref["xml"]+ref.__repr__())
            ref=BeautifulStoneSoup(fullstring).find("reference")
            processReferenceXML(ref, doc, False)

        xmltext=ref.__repr__()
        authors=ref.findAll("author")
        authorlist=[]
        surnames=[]
        original_id=ref["id"] if ref.has_key("id") else None

        if authors:
            for a in authors:
                astring=a.__repr__()
                surname=a.find("surname")
                if surname:
                    surnames.append(surname.text)
                    surname=surname.text
    ##                astring=astring.replace("<surname>","").replace("</surname>","")
                astring=cleanxml(astring)
                authorlist.append(astring)
        else:
            srnms=ref.findAll("surname")
            for s in srnms:
                surnames.append(s.text)

        if len(surnames)==0:
            for a in authorlist:
                surnames.extend(a.split())

        title=ref.find("title")
        title=title.text if title else tryToExtractTitle(xmltext)

        date=ref.find("date")
        if not date:
            match=rxsingleyear.search(xmltext)
            if match:
                date=match.group(0)
            else:
                date="????" # wooooot! no date? Must be wrong
                if ref.find("author"):
                    doc["metadata"]["ADD_NEXT_REF"]=True  # no date in this one, maybe it is in the next one, so add it
                elif len(doc["references"]) > 0 and firstcall:
    ##            if len(doc["references"]) > 0 and firstcall:
                    fuseReferences(doc, ref)
                    return

        else:
            date=date.text

        if doc["metadata"].pop("ADD_NEXT_REF", False):
            if len(doc["references"]) > 0:
                fuseReferences(doc, ref)
            return

        newref=doc.addReference()
        newref["xml"]=xmltext
        newref["text"]=cleanxml(xmltext)
        newref["authors"]=authorlist
        newref["surnames"]=surnames
        newref["title"]=title
        newref["year"]= date
        if original_id:newref["original_id"]=original_id
        return newref


    def processCitationXML(intext):
        """
            Extract the authors, date of an in-text citation <ref> from XML dom
        """
        if isinstance(intext,basestring):
            xml=BeautifulStoneSoup(intext)
        else:
            xml=intext

        if not xml:
            return None, None
        authors=[]
        for a in xml.findAll("refauthor"):
            authors.append(a.text)
        date=xml.find("date")
        if date:
            date=cleanxml(date.__repr__())
        else:
            date=""

        if authors==[] or date =="":
            return None, None
        else:
            return authors, date

    def processCitation(intext):
        """
            Extract authors and date from in-text citation using plain text and regex
        """
        authors=[]
        year=rxsingleyear.search(intext)
        if year: year=year.group(0)

        match=rxwtwoauthors.search(intext)
        if match:
            authors.append(match.group(1))
            authors.append(match.group(2))
        else:
            match=rxetal.search(intext)
            if match:
                authors.append(match.group(1))
            else: # not X and X, not et al - single author
                intext=intext.replace(","," ").replace("."," ")
                bits=intext.split()
                authors.append(bits[0])

        return authors, year

    def loadCitation(ref, sentence_id, newDocument, section):
        """
            Extract all info from <ref> tag, return dictionary
        """
        res=newDocument.addCitation()

        if ref.has_key("citation_id"):
            res["original_id"]=ref["citation_id"]

        res["original_text"]=ref.__repr__()
        res["ref_id"]=0
        res["parent_section"]=section

        if ref.has_key("refid"):
            if ref["refid"] != "?":
                replist=newDocument["metadata"].get("ref_replace_list",{})
                if str(ref["refid"]) in replist:
                    res["ref_id"]=replist[str(ref["refid"])]
                else:
                    res["ref_id"]=ref["refid"]
            else:
                # try to match citation with reference
                # don't, do it later
                pass

        authors, date=processCitationXML(ref)
        if not authors or not date:
            authors, date=processCitation(ref.__repr__())

        res["authors"]=authors
        res["date"]=date
        res["parent_s"]=sentence_id
        return res

    def findMatchingReferenceByOriginalId(id, doc):
        """
            Returns a reference from the bibliography by its original_id if found, None otherwise
        """
        for ref in doc["references"]:
            if ref.has_key("original_id") and str(ref["original_id"])==str(id):
                return ref
        return None

    def matchCitationWithReference(intext,doc):
        """
            Matches an in-text reference with the bibliography

            TODO: check this actually works
        """

        def buildBOW(ref):
            """
            """
            bow=[surname for surname in ref["surnames"]]
            for a in ref["authors"]:
                bow.extend(a.split())
            return bow

        def computeOverlap(authors, year, bow):
            """
                Returns the score of likelihood a citation points to a reference
            """
            score=0
            for i,w in enumerate(bow):
                for a in authors:
                    if w.lower()==a.lower():
                        score+=max(0.1,1-(i*0.05))
            return score

        authors, year = processCitationXML(intext)
        if not authors or not year:
            authors, year = processCitation(intext)

        yearlen=len(str(year))
        found=False

        potentials=[]
        if not found:
            for ref in doc["references"]:
                breaking=False
                bow=buildBOW(ref)
                score=computeOverlap(authors,year,bow)
                if score >= 0.3:
                    potentials.append((ref, score))

            scores=[]
            for p in potentials:
                if yearlen < 4:
                    diff=2
                else:
                    lev_diff=99
                    try:
                        y1=int(year)
                        y2=int(p[0]["year"])
                        diff=abs(y1-y2)
                    except:
                        diff=99
                        lev_diff=levenshtein(str(year).lower(),str(p[0]["year"]).lower())
                if diff <= 2 or lev_diff <=1:
                    scores.append(p)

            scores=sorted(scores,key=lambda x:x[1],reverse=True)
            if len(scores) > 0:
    ##            print "I think ", authors, year, "matches ", scores[0][0]["authors"], scores[0][0]["year"],
    ##            print "with confidence", scores[0][1]
                return scores[0][0]

        return None

    # ------------------------------------------------------------------------------
    #   Corpus reference matching functions
    # ------------------------------------------------------------------------------

    def buildHashIndex(index):
        """
            Takes the index of papers as input, outputs a dict where the keys are the hashes of each title
        """
        res={}
        for doc in index:
            title=normalizeTitle(doc["title"])
            l=res.get(title,[])
            l.append(doc)
            doc["hash"]=title
            res[title]=l
        return res

    def loadSciXML(filename):
        """
            Load a Cambridge-style SciXML

        """

        def extractSentenceText(s, newSent_id, doc):
            """
                Returns a printable representation of the sentence where all references are now placeholders with numbers
            """
            global ref_rep_count
            ref_rep_count=0

            newSent=doc.element_by_id[newSent_id]

            def repFunc(match):
                """
                """
                global ref_rep_count
                ref_rep_count+=1

                res=" <CIT ID="+str(doc.citation_by_id[newSent["citations"][ref_rep_count-1]]["id"])+" />"
                return res

            text=s.renderContents()
            text=re.sub(r"<ref.*?</ref>",repFunc,text,0,re.IGNORECASE|re.DOTALL)
            text=re.sub(r"</?refauthor>","",text,0,re.IGNORECASE|re.DOTALL)
            return text

        def loadStructureProcessPara(p, newDocument, parent):
            newPar_id=newDocument.addParagraph(parent)

            for s in p.findChildren("s"):
                newSent_id=newDocument.addSentence(newPar_id, "")
                newSent=newDocument.element_by_id[newSent_id]
                loadAttributesIfPresent(s,["ia","az","refid"],newSent)
                refs=s.findAll("ref")
                num=len(newDocument["citations"])+1
    ##            for cit in citations:
    ##                r["citation_id"]=num
    ##                num+=1
                loaded_refs=[loadCitation(r, newSent_id, newDocument, parent) for r in refs]

                newSent["citations"]=[aref["id"] for aref in loaded_refs]
                newSent["text"]=extractSentenceText(s, newSent_id, newDocument)
                newDocument.countMultiCitations(newSent) # deal with many citations within characters of each other: make them know they are a cluster TODO cluster them

            return newPar_id

        def loadStructureProcessDiv(div, newDocument):
            header=div.find("header")
            if not header:
                header_id=0
                header_text=""
            else:
                header_id=header["id"] or 0
                header_text=re.sub(r"</?header.*?>","",header.__repr__())

            newSection_id=newDocument.addSection("root",header_text, header_id)

            for p in div.findAll("p"):
                newPar_id=loadStructureProcessPara(p, newDocument, newSection_id)

        def loadMetadataIfExists(branch, key, doc):
            meta=branch.find(key)
            if meta:
                doc["metadata"][key]=meta.text

        def loadAttributesIfPresent(branch, attributes, sent):
            """
                For each element in attributes, if present in branch, it is added to sent
            """
            for a in attributes:
                if branch.has_key(a):
                    sent[a]=branch[a]

        def loadMetadata(newDocument,paper, fileno):
            """
                Does all the painful stuff of trying to recover metadata from inside a badly converted
                SciXML file
            """
            title=paper.findChildren("title")
            newDocument["metadata"]["title"]=title[0].text if len(title) > 0 else "NO TITLE"

            if fileno=="":
                fileno=paper.find("fileno").text

            newDocument["metadata"]["fileno"]=fileno

            authors=[]
            meta=soup.find("metadata")
            if not meta:
                debugAddMessage(newDocument,"error","NO METADATA IN DOCUMENT! file:"+filename)
                return newDocument

            for a in meta.findChildren("author"):
                authors.append(processPlainTextAuthor(a.text))

            if authors==[]:
                authorlist=soup.find("authorlist")

            if authorlist:
                for author in authorlist.findChildren("refauthor"):
                    authors.append(author.text)

            appeared=meta.find("appeared")
            if appeared:
                loadMetadataIfExists(appeared, "conference", newDocument)
                loadMetadataIfExists(appeared, "year", newDocument)

            newDocument["metadata"]["authors"]=authors
            newDocument["metadata"]["year"]=meta.find("year").text

        def sanitizeString(s, maxlen=200):
            s=s.replace("\t"," ")
            s=s[:maxlen]
            return s

        def makeSureValuesAreReadable(newDocument):
            newDocument["metadata"]["title"]=sanitizeString(newDocument["metadata"]["title"])
            newAuthors=[]
            for author in newDocument["metadata"]["authors"]:
                newAuthors.append(sanitizeString(author,70))
            newDocument["metadata"]["authors"]=newAuthors

            newSurnames=[]
            for surname in newDocument["metadata"]["surnames"]:
                newSurnames.append(sanitizeString(surname,25))
            newDocument["metadata"]["surnames"]=newSurnames

            newDocument["metadata"]["year"]=sanitizeString(newDocument["metadata"]["year"])
            if newDocument["metadata"].has_key("conference"):
                newDocument["metadata"]["conference"]=sanitizeString(newDocument["metadata"]["conference"])

        def matchCitationsWithReferences(newDocument):
            """
                Match each citation with its reference
            """
            allcitations=[]
            for s in newDocument.allsentences:
                for citation_id in s["citations"]:
                    cit=newDocument.citation_by_id[citation_id]

                    if cit["ref_id"] != 0: # the citation already has a matching reference id in the original document, use it
                        match=findMatchingReferenceByOriginalId(cit["ref_id"],newDocument)
                        if not match:
    ##                        print cit
                            match=newDocument.matchReferenceById(cit["ref_id"])
                    else:
                        # attempt to guess which reference the citation should point to
                        match=matchCitationWithReference(cit["original_text"],newDocument)

                    if match:
                        # whatever the previous case, make sure citation points to the ID of its reference
                        cit["ref_id"]=match["id"]
                        match["citations"].append(cit["id"]) # add the citation ID to the reference's list of citations
                        cit.pop("authors","")
                        cit.pop("date","")
                        cit.pop("original_text","")
                    else:
                        debugAddMessage(newDocument,"notes","NO MATCH for CITATION in REFERENCES: "+ cleanxml(cit["original_text"])+", ")
                        pass

        # main loadSciXML
        text=loadFileText(filename)
        soup=BeautifulStoneSoup(text)

        fileno=soup.find("docno")
        fileno=fileno.text if fileno else ""

        # Create a new SciDoc to store the paper
        newDocument=scidoc.SciDoc()
        newDocument["metadata"]["filename"]=os.path.basename(filename)
        newDocument["metadata"]["filepath"]=filename

        paper=soup.find("paper")
        if not paper:
            debugAddMessage(newDocument,"error","NO <PAPER> IN THIS PAPER! file: "+filename)
            return newDocument

        # Load metadata, either from corpus or from file
        key=cp.Corpus.getFileUID(newDocument["metadata"]["filename"])
        if cp.Corpus.metadata_index.has_key(key):
            metadata=cp.Corpus.metadata_index[key]
        else:
            metadata=None

        if metadata:
            newDocument["metadata"]["conference"]=""
            for field in metadata:
                newDocument["metadata"][field]=metadata[field]
        else:
            loadMetadata(newDocument, paper, fileno)
            debugAddMessage(newDocument,"error","PAPER NOT IN METADATA FILE! file: "+filename)

        newDocument["metadata"]["guid"]=cp.Corpus.generateGUID(newDocument["metadata"])

        # Clean up potential weird text in XML metadata
        makeSureValuesAreReadable(newDocument)

        # Load all references from the XML
        for ref in soup.findAll("reference"):
            processReferenceXML(ref, newDocument)

        # Load Abstract
        abstract=soup.find("abstract")
        if not abstract:
            debugAddMessage(newDocument,"error","CANNOT LOAD ABSTRACT! file: "+ filename+"\n")
            # TODO: LOAD first paragraph as abstract
        else:
            newSection_id=newDocument.addSection("root","Abstract")
            newPar_id=newDocument.addParagraph(newSection_id)

            for s in abstract.findChildren("a-s"):
                newSent_id=newDocument.addSentence(newPar_id, s.text)
                loadAttributesIfPresent(s,["ia","az","refid"],newDocument.element_by_id[newSent_id])

            newDocument.abstract=newDocument.element_by_id[newSection_id]

        for div in soup.findAll("div"):
            loadStructureProcessDiv(div, newDocument)

        # try to match each citation with its reference
        matchCitationsWithReferences(newDocument)

    # "in press", "forthcoming", "submitted", "to appear" = dates to fix & match
    # No functiona por: unicode
    ##    for ref in newDocument["references"]:
    ##        k=ref.get("AZ",["NO AZ"])
    ##        print k, most_common(k)

        return newDocument




def authorOverlap(authors,surnames):
    """
        Computes a 0 to 1 score for author overlap
    """
    match=0

    for s in surnames:
        for a in authors:
            au=a["family"].lower()
            s=s.lower()
            if s in au:
                match+=1
                break

    if len(surnames) > 0:
        return match/float(len(surnames))
    else:
        return 0


def matchGenericSection(header, prev):
    """
        Returns what generic section you're in
    """
    # word overlap scores?
    sections={"Introduction":"introduction",
    "Abstract":"abstract",
    "Conclusion":"conclusion concluding remarks conclusions",
    "Future work": "future future future",
    "Data":"data corpus cp statistics dataset datasets corpus corpus",
    "Related work":"previous work related works comparison",
    "Results/Discussion":"discussion discussions results",
    "Problem":"problem task",
    "Motivation":"background motivation problem formalism idea example investigating",
    "Methodology":"methods method methodology feature features algorithm heuristic our approach preprocessing model training measure measures framework",
    "Implementation":"implementation overview",
    "Experiments":"experiment experiments experimental experimentation",
    "Evaluation":"evaluation evaluating judgement analysis test performance",
    "Acknowledgements": "acknowledgements"
    }

    for s in sections:
        sections[s]=sections[s].split()

    scores={"":0}
    words=re.sub(r"[\d\.,-:;\s]{1,30}"," ",header.lower()).strip().split()
    wdict={}
    for w in words:
        wdict[w]=wdict.get(w,0)+1

    for w in wdict:
        for s in sections:
            for w2 in sections[s]:
                if w==w2:
                    scores[s]=scores.get(s,0)+wdict[w]

    res=sorted(scores.iteritems(),key=lambda x:x[1],reverse=True)[0]
    return res[0]

def matchCoreSC(section,prev):
    """
        Returns the CoreSC equivalent of the current section
    """
    CSC=['Hypothesis', 'Motivation', 'Goal', 'Object', 'Background', 'Method', 'Experiment', 'Model',
 'Observation', 'Result', 'Conclusion']
    header=section["header"].lower()
    match1=re.match(r"\s*\d\.\d(\.\d)*.*",header)
    if match1: # is subheader
        #can ignore
        pass

    for c in [c.lower() for c in CSC]:
        if c in header:
            print "yay",c+"!"

    print header

def writeTuplesToCSV(columns,tuples,filename):
    """
    """
    try:
        f=codecs.open(filename,"wb","utf-8", errors="replace")
    except:
        f=codecs.open(filename+str(random.randint(10,100)),"wb","utf-8", errors="replace")

    line=u"".join([c+u"\t" for c in columns])
    line=line.strip(u"\t")
    line+=u"\n"
    f.write(line)

    pattern=u"".join([u"%s\t" for c in columns])
    pattern=pattern.strip()
    pattern+=u"\n"

    for l in tuples:
        try:
            line=pattern % l
            f.write(line)
        except:
            print "error writing: ", l

    f.close()

def loadOrProcessSciXML(filename):
    """
    """
    doc=cp.Corpus.loadSciDoc(cp.Corpus.getFileUID(filename))
    if not doc:
        doc=loadSciXML(cp.Corpus.inputXML_dir+filename)
        ensureDirExists(cp.Corpus.jsonDocs_dir)
##        cp.Corpus.savePickledXML(filename,doc)
        cp.Corpus.saveSciDoc(doc)
##        doc.saveToFile(cp.Corpus.jsonDocs_dir+doc["guid"]+".json")
    return doc

def main():
    cp.Corpus.connectCorpus("C:\\nlp\\phd\\bob")
##    cp.Corpus.loadOldPickledIndex()
    doc=loadSciXML(cp.Corpus.inputXML_dir+"j97-1002.xml")
    doc.saveToFile("C:\\nlp\\phd\\bob\\"+doc["metadata"]["guid"]+".json")
    pass

if __name__ == '__main__':
    main()

