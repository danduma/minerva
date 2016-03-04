# functions to generate bag-of-words representations of documents for indexing
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import re, copy
from collections import defaultdict, OrderedDict
from string import punctuation

from nlp_functions import tokenizeText, stemText, stopwords, stemTokens, \
CITATION_PLACEHOLDER, unTokenize, ESTIMATED_AVERAGE_WORD_LENGTH, removeCitations, \
PAR_MARKER, CIT_MARKER, BR_MARKER, AZ_ZONES_LIST, CORESC_LIST, formatSentenceForIndexing

from minerva.proc.general_utils import getListAnyway
from minerva.az.az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST
import minerva.db.corpora as cp

# this is for adding fields to a document in Lucene. These fields are not to be indexed
FIELDS_TO_IGNORE=["left_start","right_end","params","guid_from","year_from"]


def findCitationInFullText(cit,doctext):
    """
        To avoid seeking the citation again for every query extraction method
    """
    return re.search(r"<cit\sid="+str(cit["id"])+r"\s*?/>",doctext, flags=re.DOTALL|re.IGNORECASE)


def getDictOfLuceneIndeces(prebuild_indices):
    """
        Make a simple dictionary of {method_10:{method details}}

        For some reason this function is different from getDictOfTestingMethods().
        I'm still not sure why, but apparently "az_annotated" is "standard_multi" when
        building indices but "annotated_boost" as a doc_method. ???
    """
    res=OrderedDict()
    for method in prebuild_indices:
        for parameter in prebuild_indices[method]["parameters"]:
            if prebuild_indices[method]["type"] in ["standard_multi","inlink_context"]:
                indexName=method+"_"+str(parameter)
                res[indexName]=copy.deepcopy(prebuild_indices[method])
                res[indexName]["method"]=prebuild_indices[method].get("bow_name",method)
                res[indexName]["parameter"]=parameter
                res[indexName]["index_filename"]=indexName
                res[indexName]["index_field"]=str(parameter)
                res[indexName]["options"]=prebuild_indices[method].get("options",{}) # options apply to all parameters
            elif prebuild_indices[method]["type"] in ["ilc_mashup"]:
                for ilc_parameter in prebuild_indices[method]["ilc_parameters"]:
                    indexName=method+"_"+str(parameter)+"_"+str(ilc_parameter)
                    res[indexName]=copy.deepcopy(prebuild_indices[method])
                    res[indexName]["method"]=prebuild_indices[method].get("bow_name",method)
                    res[indexName]["parameter"]=parameter
                    res[indexName]["ilc_method"]=prebuild_indices[method]["ilc_method"]
                    res[indexName]["ilc_parameter"]=ilc_parameter
                    res[indexName]["index_filename"]=indexName
                    res[indexName]["index_field"]=str(parameter)+"_"+str(ilc_parameter)
                    res[indexName]["options"]=prebuild_indices[method].get("options",{}) # options apply to all parameters
    return res


def removePunctuation(tokens):
    return [token for token in tokens if token not in punctuation]

def processContext(match,doctext, wleft, wright):
    """
        Optimize the extraction of context by limiting the amount of characters and pre-tokenizing
    """
    left_start=max(0,match.start()-((wleft+15)*ESTIMATED_AVERAGE_WORD_LENGTH))
    right_end=match.end()+((wright+15)*ESTIMATED_AVERAGE_WORD_LENGTH)

    left=doctext[left_start:match.start()] # tokenize!
    left=tokenizeText(removeCitations(left))
    left=removePunctuation(left)

    right=doctext[match.end():right_end] # tokenize!
    right=tokenizeText(removeCitations(right))
    right=removePunctuation(right)

    return {"left":left,"right":right,"left_start":left_start,"right_end":right_end}

def identifyReferenceLinkIndex(docfrom, target_guid):
    """
        Returns the id in the references list of a document that a
        "linked to" document matches

        Args:
            docfrom: full SciDoc of source of incoming link for target_guid
            target_guid: document that is cited by docfrom
    """
    for ref in docfrom["references"]:
        match=cp.Corpus.matcher.matchReference(ref)
        if match and match["guid"]==target_guid:
            return ref["id"]
    return None


def addFieldsFromDicts(d1,d2):
    """
        Concatenates the strings in dictionary d2 to those in d1,
        saves in d1
    """
    for key in d2:
        if key not in FIELDS_TO_IGNORE:
            # !TODO temporary fix to lack of separation of sentences. Remove this when done.
            d1[key]=d1.get(key,"")+" "+d2[key].replace("."," . ")
##            d1[key]=d1.get(key,"")+" "+d2[key]

def joinTogetherContextExcluding(context_list, exclude_files=[], max_year=None, exclude_authors=[], full_corpus=False):
    """
        Context_list= list of dicts{"guid_from":<guid>,"year_from":<year>,"text":<text>}}
        Concatenate all contexts, excluding the one for exclude_file
        Returns a dictionary that joins together the context as selected in one BOW
        {"inlink_context"="bla bla"}
    """
    # !TODO deal with AZ-tagged context
    # dealt with it already?
    res=defaultdict(lambda:"")

    assert isinstance(context_list,list)
    for cont in context_list:
        if cont["guid_from"] not in exclude_files:
            # !TODO Exclude authors from inlink_context and test
            if not max_year:
                addFieldsFromDicts(res,cont)
            elif int(cont["year_from"]) <= max_year:
                addFieldsFromDicts(res,cont)
        else:
##            print "Excluded", cont["guid_from"]
            pass

    # Important! RENAMING "text" field to "inlink_context" for all BOWs thus created
    res["inlink_context"]=res.pop("text") if res.has_key("text") else ""

    return res

def filterInlinkContext(context_list, exclude_list=[], max_year=None, exclude_authors=[], full_corpus=False):
    """
        Deals with the difference between a per-test-file-index and a full cp.Corpus index,
        creating the appropriate
    """
    if full_corpus:
        #!TODO FIX METHOD!
        bow={}
        for exclude_guid in exclude_list:
            bow_temp=joinTogetherContextExcluding(context_list,[exclude_guid], max_year)
            bow[getFieldSpecialTestName(method,exclude_guid)]=bow_temp[method]
        bows=[bow]
    else:
        bows=[joinTogetherContextExcluding(context_list, exclude_list, max_year)]

    return bows


# prebuilt files functions


#===============================================================================
# generateDocBOW methods = generate VSM representation of document
#===============================================================================
def generateDocBOWInlinkContext(doc_incoming, parameters, doctext=None):
    """
        Create a BOW from all the inlink contexts of a given document

        :param doc_incoming: which document to generate inlink_context BOW for.
        :param parameters: iterable with
    """
    all_contexts={}
    doc_incoming_guid=doc_incoming.metadata["guid"]
    for param in parameters:
        all_contexts[param]=[]

    doc_metadata=cp.Corpus.getMetadataByGUID(doc_incoming_guid)
    print("Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]), "incoming links")

    for inlink_guid in doc_metadata["inlinks"]:
        # loads from cache if exists, XML otherwise
        docfrom=cp.Corpus.loadSciDoc(inlink_guid)
        # important! the doctext here has to be that of the docfrom, NOT doc_incoming
        doctext=docfrom.getFullDocumentText()
        ref_id=identifyReferenceLinkIndex(docfrom, doc_incoming_guid)
        if not ref_id:
            print("ERROR: Cannot match in-collection document %s with reference in file %s" %())

        print("Document with incoming citation links loaded:", docfrom["metadata"]["filename"])

        for param in parameters:
            all_contexts[param]=all_contexts.get(param,[])

            for cit in docfrom["citations"]:
                if cit["ref_id"] == ref_id:
                    # need a list of citations for each outlink, to extract context from each
                    match=findCitationInFullText(cit,doctext)
                    if not match:
                        print("Weird! can't find citation in text")
                        continue

                    context=getOutlinkContextWindowAroundCitation(match, doctext, param, param)
                    # code below: store list of dicts with filename, BOW, so I can filter out same file later
                    context["guid_from"]=docfrom["metadata"]["guid"]
                    context["year_from"]=docfrom["metadata"]["year"]
                    all_contexts[param].append(context)

    #   this bit of code makes every entry a list for multiple representations from each document
##    for c in all_contexts:
##        all_contexts[c]=[all_contexts[c]]
    return all_contexts


def generateDocBOW_ILC_Annotated(doc_incoming, parameters, doctext=None, full_paragraph=True):
    """
        Create a BOW from all the inlink contexts of a given document
        Extracts sentences around the citation, annotated with their AZ

        Args:
            doc_incoming: for compatibility, SciDoc or dict with .metadata["guid"]
            parameters = {"full_paragraph":True,"sent_left":1, "sent_right":1}?
    """
    doc_incoming_guid=doc_incoming.metadata["guid"]
    all_contexts=defaultdict(lambda:[])
    for param in parameters:
        all_contexts[param]=[]

    doc_metadata=cp.Corpus.getMetadataByGUID(doc_incoming_guid)
    print("Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]), "incoming links")

    for inlink_guid in doc_metadata["inlinks"]:
        # loads from cache if exists, XML otherwise
        docfrom=cp.Corpus.loadSciDoc(inlink_guid)
        cp.Corpus.annotateDoc(docfrom,["AZ"])

        # important! the doctext here has to be that of the docfrom, NOT doc_incoming
        doctext=docfrom.getFullDocumentText()
        ref_id=identifyReferenceLinkIndex(docfrom, doc_incoming_guid)

        print("Document with incoming citation links loaded:", docfrom["metadata"]["filename"])

        for param in parameters:
            citations=[cit for cit in docfrom["citations"] if cit["ref_id"] == ref_id]
            for cit in citations:
                to_add=selectSentencesToAdd(docfrom,cit,param)

                context={"ilc_AZ_"+zone:"" for zone in AZ_ZONES_LIST}
                for zone in CORESC_LIST:
                    context["ilc_CSC_"+zone]=""
                to_add=[]

                for sent_id in to_add:
                    sent=docfrom.element_by_id[sent_id]
                    text=formatSentenceForIndexing(sent)
                    context["ilc_AZ_"+sent["az"]]+=" "+text
                    if "csc_type" not in sent:
                        sent["csc_type"]="Bac"
                    context["ilc_CSC_"+sent["csc_type"]]+=" "+text

                context["guid_from"]=docfrom["metadata"]["guid"]
                context["year_from"]=docfrom["metadata"]["year"]
                all_contexts[param].append(context)

    #   this bit of code makes every entry a list for multiple representations from each document
##    for c in all_contexts:
##        all_contexts[c]=[all_contexts[c]]
    return all_contexts


def mashupBOWinlinkMethods(doc_incoming, exclude_files, max_year, method_params, full_corpus=False):
    """
        Returns BOWs ready to add to indeces of different parameters for inlink_context
    """
    doc_incoming_guid=doc_incoming.metadata["guid"]
    ilc_bow=cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, method_params["ilc_method"], method_params["ilc_parameter"])
    if ilc_bow is None:
        print("Prebuilt BOW not found for: inlink_context", method_params["ilc_method"] + method_params["ilc_parameter"])
        return None

    # this is a dict representing the context and eventually a new "document" to be added to the index
    ilc_bow=filterInlinkContext(ilc_bow, exclude_files, max_year, full_corpus=full_corpus)

    # for some reason this shouldn't be a list
    ilc_bow=ilc_bow[0]

    bow_method1=getListAnyway(cp.Corpus.loadPrebuiltBOW(doc_incoming_guid,method_params["mashup_method"],method_params["parameter"]))

    res=[]
    for passage in bow_method1:
        addFieldsFromDicts(passage,ilc_bow)
##        addDocBOWFullTextField(doc,res,doctext)
        res.append(passage)
    return res

def mashupAnnotatedBOWinlinkMethods(doc_incoming, exclude_files, max_year, method1, param1, inlink_parameter):
    """
        Returns BOWs ready to add to indeces of different parameters for inlink_context
    """
    doc_incoming_guid=doc_incoming.metadata["guid"]
    ilc_bow=cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, "inlink_context", inlink_parameter)
    if ilc_bow is None:
        print("Prebuilt BOW not found for: inlink_context", inlink_parameter)
        return None

    # this is a dict representing the context and eventually a new "document" to be added to the index
    ilc_bow=joinTogetherContextExcluding(ilc_bow,exclude_files,max_year)

    bow_method1=getListAnyway(cp.Corpus.loadPrebuiltBOW(doc_incoming_guid,method1,param1))

    res=[]
    for passage in bow_method1:
        addFieldsFromDicts(passage,ilc_bow)
##        addDocBOWFullTextField(doc,res,doctext)

        res.append(passage)
    return res


def getDocBOWInlinkContextCache(doc_incoming, exclude_files, parameters, doctext=None):
    """
        Same as getDocBOWInlinkContext, uses prebuilt inlink contexts from disk cache,
        rebuilds them where not available
    """
    doc_incoming_guid=doc_incoming.metadata["guid"]
    newparam=[]
    result={}
##    self_id=doc["guid"] # avoid including in the BOW context from the very file
##    max_date=doc["year"]

    for param in parameters:

        if not cp.Corpus.cachedJsonExists("bow",doc_incoming_guid,{"method":"inlink_context", "parameter":param}):
            newparam.append(param)
        else:
##            print "Loading prebuilt inlink_context BOW from cache for", doc_incoming["filename"], param
            bow=cp.Corpus.loadPrebuiltBOW(doc_incoming_guid,"inlink_context", param)
            # bow[0] would be necessary if the function returns a list of BOWs for each param

            result[param]=[{"text":joinTogetherContextExcluding(bow,exclude_files,)}]

    if len(newparam) > 0:
        print("New parameters I don't have prebuilt BOWs for:",newparam)
        new=generateDocBOWInlinkContext(doc_incoming, newparam, doctext)
        for param in new:
            result[param]=[{"text":joinTogetherContextExcluding(new[param],exclude_files)}]

    return result


def getDocBOWfull(doc, parameters=None, doctext=None):
    """
        Get BOW for document using full text minus references and section titles

        Args:
            doc: full SciDoc to get text for
    """
    if not doctext:
        doctext=doc.getFullDocumentText(doc)
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    new_doc={"text":unTokenize(tokens)}
    return {1:[new_doc]} # all functions must take a list of parameters and return dict[parameter]=list of BOWs

def getDocBOWpassagesMulti(doc, parameters=[100], doctext=None):
    """
        Get BOW for document using full text minus references and section titles

        Args:
            doc: full SciDoc to get text for
        Returns:
             multiple BOWs in a dictionary where the keys are the parameters
    """

    if not doctext:
        doctext=doc.getFullDocumentText(doc, headers=True)

    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    res={}

    for param in parameters:
        res[param]=[]

        for i in xrange(0, len(tokens), param):
            bow={"text":unTokenize(tokens[i:i+param])}
            res[param].append(bow)

        for i in xrange(param/2, len(tokens), param):
            bow={"text":unTokenize(tokens[i:i+param])}
            res[param].append(bow)

    return res

def getDocBOWTitleAbstract(doc, parameters=None, doctext=None):
    """
        Get BOW for document made up of only title and abstract
    """
    paragraphs=[]
    doctext=doc["metadata"]["title"]+". "
    if len(doc.allsections) > 0:
        try:
            doctext+=doc.getSectionText(doc.allsections[0],False)
        except:
            doctext+=u"<UNICODE ERROR>"
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    return {1:[{"text":unTokenize(tokens)}]}

def getDocBOWannotated(doc, parameters=None, doctext=None, keys=["az","csc_type"]):
    """
        Get BOW for document with AZ/CSC
    """
    res=defaultdict(lambda:"")
    for sentence in doc.allsentences:
        text=formatSentenceForIndexing(sentence)
        for key in keys:
            if sentence.has_key(key):
                res[sentence[key]]+=text
    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}

def getDocBOWrandomZoning(doc, parameters=None, doctext=None, keys=["az","csc_type"]):
    """
        Get BOW for document with randomized AZ/CSC
    """
    res=defaultdict(lambda:"")
    for sentence in doc.allsentences:
        text=formatSentenceForIndexing(sentence)
        res[random.choice(AZ_ZONES_LIST)]+=text
        res[random.choice(CORESC_LIST)]+=text

    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}


def getDocBOWannotatedSections(doc, parameters=None, doctext=None):
    """
        Returns a dict where each key should be a field for the document in
        the Lucene index
        returns {"title","abstract","text"}
    """
    paragraphs=[]
    res={}
    res["title"]=doc["metadata"]["title"]
    res["abstract"]=""

    if len(doc.allsections) > 0 :
        try:
            res["abstract"]=doc.getSectionText(doc.allsections[0],False)
            res["abstract"]==removeCitations(res["abstract"]).lower()
        except:
            res["abstract"]=u"<UNICODE ERROR>"

    paper_text=""
    for index in range(1,len(doc.allsections),1):
        paper_text+=doc.getSectionText(doc.allsections[index],False)

    res["text"]=removeCitations(paper_text).lower()
    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}

def addDocBOWFullTextField(doc,res_dict,doctext=None):
    """
        Adds the _full_text field
    """
    if not doctext:
        doctext=doc.getFullDocumentText(doc)
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    res_dict["_full_text"]=unTokenize(tokens)

def main():

    pass

if __name__ == '__main__':
    main()
