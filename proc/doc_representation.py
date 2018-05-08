# functions to generate bag-of-words representations of documents for indexing
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import re, copy
from collections import defaultdict, OrderedDict
from string import punctuation

from .nlp_functions import (tokenizeText, stemText, stopwords, stemTokens,
                            CITATION_PLACEHOLDER, unTokenize, ESTIMATED_AVERAGE_WORD_LENGTH, removeCitations,
                            PAR_MARKER, CIT_MARKER, BR_MARKER, AZ_ZONES_LIST, CORESC_LIST,
                            formatSentenceForIndexing, selectSentencesToAdd)
from proc.query_extraction import WindowQueryExtractor, SentenceQueryExtractor
from proc.general_utils import getListAnyway
from importing.fix_citations import fixDocRemovedCitations
from az.az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST
import db.corpora as cp
from db.base_corpus import shouldIgnoreCitation
from six.moves import range
import random

# this is for adding fields to a document in Lucene. These fields are not to be indexed
# FIELDS_TO_IGNORE = ["left_start", "right_end", "params", "guid_from", "year_from",
#                     "authors_from", "authors_to", "guid_to", "year_to"]

FIELDS_TO_IGNORE = ["left_start", "right_end", "params", "guid_from", "year_from",
                    "authors_from", "authors_to", "guid_to", "year_to", "_metadata"]


def findCitationInFullTextXML(cit, doctext):
    """
        To avoid seeking the citation again for every query extraction method
    """
    return re.search(r"<cit\sid=" + str(cit["id"]) + r"\s*?/>", doctext, flags=re.DOTALL | re.IGNORECASE)


def findCitationInFullTextUnderscores(cit, doctext):
    """
        To avoid seeking the citation again for every query extraction method
    """
    return re.search(r"\_\_" + str(cit["id"]).lower(), doctext, flags=re.DOTALL | re.IGNORECASE)


def getDictOfLuceneIndeces(prebuild_indices):
    """
        Make a simple dictionary of {method_10:{method details}}

        For some reason this function is different from getDictOfTestingMethods().
        I'm still not sure why, but apparently "az_annotated" is "standard_multi" when
        building indices but "annotated_boost" as a doc_method. ???
    """
    res = OrderedDict()
    for method in prebuild_indices:
        for parameter in prebuild_indices[method]["parameters"]:
            if prebuild_indices[method]["type"] in ["standard_multi", "inlink_context"]:
                indexName = method + "_" + str(parameter)
                res[indexName] = copy.deepcopy(prebuild_indices[method])
                res[indexName]["method"] = prebuild_indices[method].get("bow_name", method)
                res[indexName]["parameter"] = parameter
                res[indexName]["index_filename"] = indexName
                res[indexName]["index_field"] = str(parameter)
                res[indexName]["options"] = prebuild_indices[method].get("options",
                                                                         {})  # options apply to all parameters
            elif prebuild_indices[method]["type"] in ["ilc_mashup"]:
                for ilc_parameter in prebuild_indices[method]["ilc_parameters"]:
                    indexName = method + "_" + str(parameter) + "_" + str(ilc_parameter)
                    res[indexName] = copy.deepcopy(prebuild_indices[method])
                    res[indexName]["method"] = prebuild_indices[method].get("bow_name", method)
                    res[indexName]["parameter"] = parameter
                    res[indexName]["ilc_method"] = prebuild_indices[method]["ilc_method"]
                    res[indexName]["ilc_parameter"] = ilc_parameter
                    res[indexName]["index_filename"] = indexName
                    res[indexName]["index_field"] = str(parameter) + "_" + str(ilc_parameter)
                    res[indexName]["options"] = prebuild_indices[method].get("options",
                                                                             {})  # options apply to all parameters
    return res


def removePunctuation(tokens):
    return [token for token in tokens if token not in punctuation]


def processContext(match, doctext, wleft, wright):
    """
        Optimize the extraction of context by limiting the amount of characters and pre-tokenizing
    """
    left_start = max(0, match.start() - ((wleft + 15) * ESTIMATED_AVERAGE_WORD_LENGTH))
    right_end = match.end() + ((wright + 15) * ESTIMATED_AVERAGE_WORD_LENGTH)

    left = doctext[left_start:match.start()]  # tokenize!
    left = tokenizeText(removeCitations(left))
    left = removePunctuation(left)

    right = doctext[match.end():right_end]  # tokenize!
    right = tokenizeText(removeCitations(right))
    right = removePunctuation(right)

    return {"left": left, "right": right, "left_start": left_start, "right_end": right_end}


def identifyReferenceLinkIndex(docfrom, target_guid):
    """
        Returns the id in the references list of a document that a
        "linked to" document matches

        Args:
            docfrom: full SciDoc of source of incoming link for target_guid
            target_guid: document that is cited by docfrom
    """
    for ref in docfrom["references"]:
        match = cp.Corpus.matcher.matchReference(ref)
        if match and match["guid"] == target_guid:
            return ref["id"]
    return None


def addFieldsFromDicts(d1, d2):
    """
        Concatenates the strings in dictionary d2 to those in d1,
        saves in d1
    """
    for key in d2:
        if key not in FIELDS_TO_IGNORE:
            # !TODO temporary fix to lack of separation of sentences. Remove this when done.
            d1[key] = d1.get(key, "") + " " + d2[key].replace(".", " . ")


##            d1[key]=d1.get(key,"")+" "+d2[key]

def joinTogetherContextExcluding(context_list, exclude_files=[], filter_options={}):
    """
        Concatenate all contexts, excluding the ones for guids in exclude_files and also ignoring
        whatever matches the filter_options using shouldIgnoreCitation()

        :param context_list: list of dicts{"guid_from":<guid>, "year_from":<year>, "text":<text>}}
        :param exclude_files: list of files to not get the context from
        :returns: Returns a dictionary that joins together the contexts as selected in one BOW
                    {"inlink_context"="bla bla"}
    """
    # !TODO deal with AZ-tagged context
    # dealt with it already?
    res = defaultdict(lambda: "")

    assert isinstance(context_list, list)
    for cont in context_list:
        metadata = cont["_metadata"]
        if metadata["guid_from"] not in exclude_files:
            # if "authors_from" in metadata:
            #     meta_from = {"authors": metadata["authors_from"],
            #                  "year": metadata["year_from"],
            #                  "guid": metadata["guid_from"]}
            #     meta_to = {"authors": metadata["authors_to"],
            #                "year": metadata["year_to"],
            #                "guid": metadata["guid_to"]}
            # else:
            #     meta_from = cp.Corpus.getMetadataByGUID(metadata["guid_from"])
            #     meta_to = cp.Corpus.getMetadataByGUID(metadata["guid_to"])
            #
            # if shouldIgnoreCitation(meta_from, meta_to, filter_options):
            #     # print("Ignored citation: ", metadata)
            #     continue

            addFieldsFromDicts(res, cont)
        else:
            ##            print "Excluded", cont["guid_from"]
            pass

    # Important! RENAMING "text" field to "inlink_context" for all BOWs thus created
    res["inlink_context"] = res.pop("text") if "text" in res else ""

    return res


def filterInlinkContext(context_list, exclude_list=[], full_corpus=False, filter_options={}):
    """
        Deals with the difference between a per-test-file-index and a full cp.Corpus index,
        creating the appropriate
    """
    ##    if full_corpus:
    ##        bow={}
    ##        for exclude_guid in exclude_list:
    ##            bow_temp=joinTogetherContextExcluding(context_list,[exclude_guid], max_year)
    ##            bow[getFieldSpecialTestName(method,exclude_guid)]=bow_temp[method]
    ##        bows=[bow]
    ##    else:
    ##        bows=[joinTogetherContextExcluding(context_list, exclude_list, max_year)]
    # TODO exclude authors: take out of this method? Do it before, assume the exclude_list already contains this filtering?
    bows = [joinTogetherContextExcluding(context_list,
                                         exclude_list,
                                         filter_options=filter_options)]
    return bows


def getOutlinkContextWindowAroundCitation(match, doctext, words_left, words_right):
    """

    :param match:
    :param doctext:
    :param words_left:
    :param words_right:
    :return:
    """


# ===============================================================================
# generateDocBOW methods = generate VSM representation of document
# ===============================================================================

def addILCMetadata(context, docfrom, doc_target):
    metadata = {"guid_from": docfrom["metadata"]["guid"],
                "guid_to": doc_target["metadata"]["guid"],
                "year_from": docfrom["metadata"]["year"],
                "year_to": doc_target["metadata"]["year"],
                "authors_from": docfrom["metadata"]["authors"],
                "authors_to": doc_target["metadata"]["authors"]}

    context["_metadata"] = metadata
    return context


def extractILCWindow(docfrom, doc_target, ref_id, doctext, parameters, all_contexts):
    """
    This function deals with extracting ILC using window-of-tokens.
    Returns list of contexts it's extracted.

    :param docfrom:
    :param doc_target:
    :param ref_id:
    :param parameters:
    :param all_contexts:
    :return:
    """
    for cit in docfrom["citations"]:
        if cit["ref_id"] == ref_id:
            # need a list of citations for each outlink, to extract context from each
            match = findCitationInFullTextXML(cit, doctext)
            if not match:
                match = findCitationInFullTextUnderscores(cit, doctext)
            if not match:
                print("Weird! can't find citation in text!", cit)
                print("Fixing document ", docfrom["metadata"]["guid"])
                fixDocRemovedCitations(docfrom)
                doctext = docfrom.formatTextForExtraction(docfrom.getFullDocumentText())
                match = findCitationInFullTextXML(cit, doctext)
                if not match:
                    match = findCitationInFullTextUnderscores(cit, doctext)
                if not match:
                    print("Failed to fix for this citation")
                    continue
                else:
                    cp.Corpus.saveSciDoc(docfrom)

            extractor = WindowQueryExtractor()
            params = {"parameters": parameters,
                      "match_start": match.start(),
                      "match_end": match.end(),
                      "doctext": doctext,
                      "extract_only_plain_text": True,
                      }
            contexts = extractor.extractMulti(params)

            # code below: store guid, year and authors so we can filter out contexts later
            for generated_context in contexts:
                context = addILCMetadata({"text": generated_context["text"]}, docfrom, doc_target)
                all_contexts[generated_context["params"][0]].append(context)  # ["params"][0] is wleft


def extractILCSentences(docfrom, doc_target, ref_id, parameters, all_contexts):
    """
    This deals with extracting ILC as window-of-sentences

    :param docfrom:
    :param doc_target:
    :return:
    """
    for cit in docfrom["citations"]:
        if cit["ref_id"] == ref_id:
            extractor = SentenceQueryExtractor()
            params = {"parameters": parameters,
                      "docfrom": docfrom,
                      "cit": cit,
                      "dict_key": "text",
                      "extract_only_plain_text": True,
                      }
            contexts = extractor.extractMulti(params)
            for generated_context in contexts:
                context = addILCMetadata({"text": generated_context["text"]}, docfrom, doc_target)
                all_contexts[generated_context["params"][0]].append(context)  # ["params"][0] is wleft


def generateDocBOWInlinkContext(doc_target, parameters, doctext=None, filter_options={}):
    """
        Create a BOW from all the inlink contexts of a given document.

        Now with extra filtering! It's much faster to filter out the whole document here
        rather than first generate the contexts and then have to filter them. That was
        very dumb.

        :param doc_target: which document to generate inlink_context BOW for.
        :param parameters: iterable with parameters for ILC. Either numbers or strings
        :param doctext: the text of the document. Unused, here for compatibility.
        :param filter_options: dict with options for filtering
    """
    all_contexts = {}
    doc_target_guid = doc_target.metadata["guid"]
    for param in parameters:
        all_contexts[param] = []

    window_parameters = [param for param in parameters if not isinstance(param, str)]
    sentence_parameters = [param for param in parameters if isinstance(param, str)]

    doc_metadata = cp.Corpus.getMetadataByGUID(doc_target_guid)
    # print("Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]),
    #       "incoming links")

    set_inlinks = set(doc_metadata["inlinks"])
    if len(doc_metadata["inlinks"]) > len(set_inlinks):
        print("ARGH I listed the inlinks more than once!")

    meta_to = doc_target.metadata
    for inlink_guid in doc_metadata["inlinks"]:
        meta_from = cp.Corpus.getMetadataByGUID(inlink_guid)
        if shouldIgnoreCitation(meta_from, meta_to, filter_options):
            continue

        # loads from cache if exists, XML otherwise
        docfrom = cp.Corpus.loadSciDoc(inlink_guid)
        # important! the doctext here has to be that of the docfrom, NOT doc_incoming
        doctext = docfrom.formatTextForExtraction(docfrom.getFullDocumentText())
        ref_id = identifyReferenceLinkIndex(docfrom, doc_target_guid)
        if not ref_id:
            print("ERROR: Cannot match in-collection document %s with reference in file %s" % (
                doc_target.metadata["guid"], docfrom.metadata["guid"]))

        # print("Document with incoming citation links loaded:", docfrom["metadata"]["filename"])

        for param in parameters:
            all_contexts[param] = all_contexts.get(param, [])

        if len(window_parameters) > 0:
            extractILCWindow(docfrom, doc_target, ref_id, doctext, window_parameters, all_contexts)

    return all_contexts


def generateDocBOW_ILC_Annotated(doc_target, parameters, doctext=None, filter_options={}, force_rebuild=False):
    """
        Create a BOW from all the inlink contexts of a given document.
        Extracts SENTENCES around the citation, annotated with their AZ/CSC

        :param doc_target: for compatibility, SciDoc or dict with .metadata["guid"]
        :param parameters: {"full_paragraph":True,"sent_left":1, "sent_right":1}?
        :param doctext: text of the target document. For compatibility, this function doesn't care.
        :param filter_options: dict with options for incoming links to ignore by year, author, etc.
        :param force_rebuild: always rebuilds, this parameter has no effect
        :return: list of contexts. Each context is a dict.
    """
    doc_target_guid = doc_target.metadata["guid"]
    all_contexts = defaultdict(lambda: [])
    for param in parameters:
        all_contexts[param] = []

    doc_metadata = cp.Corpus.getMetadataByGUID(doc_target_guid)
    print("Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]),
          "incoming links")

    for inlink_guid in doc_metadata["inlinks"]:
        incoming_citation_metadata = cp.Corpus.getMetadataByGUID(inlink_guid)
        if shouldIgnoreCitation(incoming_citation_metadata, doc_metadata, filter_options):
            continue

        # loads from cache if exists, XML otherwise
        docfrom = cp.Corpus.loadSciDoc(inlink_guid)
        ##        cp.Corpus.annotateDoc(docfrom,["AZ"])

        ref_id = identifyReferenceLinkIndex(docfrom, doc_target_guid)

        print("Document with incoming citation links loaded:", docfrom["metadata"]["filename"])

        for param in parameters:
            citations = [cit for cit in docfrom["citations"] if cit["ref_id"] == ref_id]
            for cit in citations:
                to_add = selectSentencesToAdd(docfrom, cit, param)

                context = {"ilc_AZ_" + zone: "" for zone in AZ_ZONES_LIST}
                for zone in CORESC_LIST:
                    context["ilc_CSC_" + zone] = ""

                for sent_id in to_add:
                    sent = docfrom.element_by_id[sent_id]
                    text = formatSentenceForIndexing(sent)
                    if sent.get("az", "") != "":
                        context["ilc_AZ_" + sent.get("az", "")] += " " + text

                    if "csc_type" not in sent:
                        sent["csc_type"] = "Bac"
                    context["ilc_CSC_" + sent["csc_type"]] += " " + text

                meta = {"guid_from": docfrom["metadata"]["guid"],
                        "guid_to": doc_target["metadata"]["guid"],
                        "year_from": docfrom["metadata"]["year"],
                        "year_to": doc_target["metadata"]["year"],
                        "authors_from": docfrom["metadata"]["authors"],
                        "authors_to": doc_target["metadata"]["authors"]}
                context["_metadata"] = meta
                all_contexts[param].append(context)

    return all_contexts


def mashupBOWinlinkMethods(doc_incoming_guid, exclude_files, index_max_year, method_params, full_corpus=False,
                           filter_options={}, force_rebuild=False):
    """
        Returns BOWs ready to add to indeces of different parameters for inlink_context
    """
    ilc_bow = cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, {"method": method_params["ilc_method"],
                                                            "parameter": method_params["ilc_parameter"]})
    if ilc_bow is None:
        print("Prebuilt BOW not found for: inlink_context",
              method_params["ilc_method"] + method_params["ilc_parameter"])
        return None

    # this is a dict representing the context and eventually a new "document" to be added to the index
    if "_metadata" in ilc_bow:
        ilc_bow = filterInlinkContext(ilc_bow, exclude_files, full_corpus=full_corpus, filter_options=filter_options)

    # for some reason this shouldn't be a list
    ilc_bow = ilc_bow[0]

    bow_method1 = getListAnyway(cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, {"method": method_params["mashup_method"],
                                                                              "parameter": method_params["parameter"]}))

    res = []
    for passage in bow_method1:
        addFieldsFromDicts(passage, ilc_bow)
        ##        addDocBOWFullTextField(doc,res,doctext)
        res.append(passage)
    return res


# TODO: ever used?
# def mashupAnnotatedBOWinlinkMethods(doc_incoming, exclude_files, filter_options, method1, param1, inlink_parameter):
#     """
#         Returns BOWs ready to add to indeces of different parameters for inlink_context
#     """
#     doc_incoming_guid = doc_incoming.metadata["guid"]
#     ilc_bow = cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, "inlink_context", inlink_parameter)
#     if ilc_bow is None:
#         print("Prebuilt BOW not found for: inlink_context", inlink_parameter)
#         return None
#
#     # this is a dict representing the context and eventually a new "document" to be added to the index
#     ilc_bow = joinTogetherContextExcluding(ilc_bow, exclude_files, filter_options=filter_options)
#
#     bow_method1 = getListAnyway(cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, method1, param1))
#
#     res = []
#     for passage in bow_method1:
#         addFieldsFromDicts(passage, ilc_bow)
#         ##        addDocBOWFullTextField(doc,res,doctext)
#
#         res.append(passage)
#     return res

def getJustTextFromILCBOW(bow):
    return bow["inlink_context"]


def getDocBOWInlinkContextCache(doc_incoming, parameters, doctext=None, exclude_files=[], filter_options={},
                                force_rebuild=False):
    """
        Same as generateDocBOWInlinkContext, uses prebuilt inlink contexts from disk
        cache, rebuilds them where not available. Can be forced to recreate them using
         "force_rebuild"
    """
    doc_incoming_guid = doc_incoming.metadata["guid"]
    newparam = []
    result = {}
    ##    self_id=doc["guid"] # avoid including in the BOW context from the very file
    ##    max_date=doc["year"]

    for param in parameters:

        if force_rebuild or not cp.Corpus.cachedJsonExists("bow", doc_incoming_guid,
                                                           {"method": "inlink_context",
                                                            "parameter": param}):
            newparam.append(param)
        else:
            bow = cp.Corpus.loadPrebuiltBOW(doc_incoming_guid, {"method": "inlink_context",
                                                                "parameter": param})
            # bow[0] would be necessary if the function returns a list of BOWs for each param

            # FIXME: DO I STILL NEED THIS?
            joined_context = joinTogetherContextExcluding(bow,
                                                          exclude_files,
                                                          filter_options=filter_options)
            result[param] = [{"text": getJustTextFromILCBOW(joined_context)}]

    if not force_rebuild and len(newparam) > 0:
        print("New parameters I don't have prebuilt BOWs for:", newparam)

    new = generateDocBOWInlinkContext(doc_incoming, newparam, doctext, filter_options=filter_options)
    for param in new:
        ilc_text = getJustTextFromILCBOW(joinTogetherContextExcluding(new[param],
                                                                      exclude_files,
                                                                      filter_options=filter_options))
        result[param] = [{"text": ilc_text}]

    return result


def getDocBOWfull(doc, parameters=None, doctext=None, filter_options={}, force_rebuild=False):
    """
        Get BOW for document using full text minus references and section titles

        Args:
            doc: full SciDoc to get text for
    """
    if not doctext:
        doctext = doc.formatTextForExtraction(doc.getFullDocumentText(doc))
    doctext = removeCitations(doctext).lower()
    tokens = tokenizeText(doctext)
    new_doc = {"text": unTokenize(tokens)}
    return {1: [new_doc]}  # all functions must take a list of parameters and return dict[parameter]=list of BOWs


def getDocBOWpassagesMulti(doc, parameters=[100], doctext=None):
    """
        Get BOW for document using full text minus references and section titles

        Args:
            doc: full SciDoc to get text for
        Returns:
             multiple BOWs in a dictionary where the keys are the parameters
    """

    if not doctext:
        doctext = doc.formatTextForExtraction(doc.getFullDocumentText(doc, headers=True))

    doctext = removeCitations(doctext).lower()
    tokens = tokenizeText(doctext)
    res = {}

    for param in parameters:
        res[param] = []

        for i in range(0, len(tokens), param):
            bow = {"text": unTokenize(tokens[i:i + param])}
            res[param].append(bow)

        for i in range(param / 2, len(tokens), param):
            bow = {"text": unTokenize(tokens[i:i + param])}
            res[param].append(bow)

    return res


def getDocBOWTitleAbstract(doc, parameters=None, doctext=None):
    """
        Get BOW for document made up of only title and abstract
    """
    paragraphs = []
    doctext = doc["metadata"]["title"] + ". "
    if len(doc.allsections) > 0:
        try:
            doctext += " " + doc.getSectionText(doc.allsections[0], False)
        except:
            doctext += u"<UNICODE ERROR>"
    doctext = removeCitations(doctext).lower()
    tokens = tokenizeText(doctext)
    return {1: [{"text": unTokenize(tokens)}]}


def getDocBOWannotated(doc, parameters=None, doctext=None, keys=["az", "csc_type"]):
    """
        Get BOW for document with AZ/CSC
    """
    res = defaultdict(lambda: [])
    for sentence in doc.allsentences:
        text = formatSentenceForIndexing(sentence)
        for key in keys:
            if key in sentence:
                res[sentence[key]].append(text)
    for key in res:
        res[key] = " ".join(res[key])
    addDocBOWFullTextField(doc, res, doctext)
    return {1: [res]}


def getDocBOWrandomZoning(doc, parameters=None, doctext=None, keys=["az", "csc_type"]):
    """
        Get BOW for document with randomized AZ/CSC
    """
    res = defaultdict(lambda: "")
    for sentence in doc.allsentences:
        text = formatSentenceForIndexing(sentence)
        res[random.choice(AZ_ZONES_LIST)] += " " + text
        res[random.choice(CORESC_LIST)] += " " + text

    addDocBOWFullTextField(doc, res, doctext)
    return {1: [res]}


def getDocBOWannotatedSections(doc, parameters=None, doctext=None):
    """
        Returns a dict where each key should be a field for the document in
        the Lucene index
        returns {"title","abstract","text"}
    """
    paragraphs = []
    res = {}
    res["title"] = doc["metadata"]["title"]
    res["abstract"] = ""

    if len(doc.allsections) > 0:
        try:
            res["abstract"] = doc.getSectionText(doc.allsections[0], False)
            res["abstract"] = removeCitations(res["abstract"]).lower()
        except:
            res["abstract"] = u"<UNICODE ERROR>"

    paper_text = ""
    for index in range(1, len(doc.allsections), 1):
        paper_text += " " + doc.getSectionText(doc.allsections[index], False)

    res["text"] = removeCitations(paper_text).lower()
    addDocBOWFullTextField(doc, res, doctext)
    return {1: [res]}


def addDocBOWFullTextField(doc, res_dict, doctext=None):
    """
        Adds the _full_text field
    """
    if not doctext:
        doctext = doc.formatTextForExtraction(doc.getFullDocumentText(doc))
    doctext = removeCitations(doctext).lower()
    tokens = tokenizeText(doctext)
    res_dict["_full_text"] = unTokenize(tokens)


def main():
    pass


if __name__ == '__main__':
    main()
