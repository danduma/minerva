# functions to add prebuilt BOWs to index
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import logging
from copy import deepcopy

import minerva.db.corpora as cp
import minerva.proc.doc_representation as doc_representation
from elastic_writer import ElasticWriter
from minerva.evaluation.prebuild_functions import prebuildMulti

def defaultAddDocument(writer, new_doc, metadata, fields_to_process, bow_info):
    """
        Add a document to the index.

        :param new_doc: dict of fields with values
        :type new_doc:dict
        :param metadata: ditto
        :type metadata:dict
        :param fields_to_process: only add these fields from the doc dict
        :type fields_to_process:list
    """

    body={"guid": metadata["guid"],
          "metadata": metadata,
          "bow_info": bow_info,
          }

    for field in fields_to_process:
        body[field]=new_doc[field]

    writer.addDocument(body)

ADD_DOCUMENT_FUNCTION=defaultAddDocument

def addBOWsToIndex(guid, indexNames, index_max_year, fwriters=None):
    """
        For one guid, add all its BOWs to the given index

        :param guid: guid of the paper
        :param indexNames: a fully expanded dict of doc_methods
        :param index_max_year: the max year to accept to add a file to the index
    """
    meta=cp.Corpus.getMetadataByGUID(guid)
    if not meta:
        logging.error("Error: can't load metadata for paper %s" % guid)
        return

    if not fwriters:
        fwriters=[]
        for indexName in indexNames:
            actual_dir=cp.Corpus.getRetrievalIndexPath(guid, indexName, full_corpus=False)
            fwriters[indexName]=ElasticWriter(actual_dir,cp.Corpus.es)

    for indexName in indexNames:
        index_data=indexNames[indexName]
        method=index_data["method"]
        parameter=index_data["parameter"]
        ilc_parameter=index_data.get("ilc_parameter","")

        if index_data["type"] in ["standard_multi"]: # annotated_boost?
            if index_max_year:
                if int(meta["year"]) > int(index_max_year):
                    continue
            addOrBuildBOWToIndex(fwriters[indexName], guid, index_data)
        elif index_data["type"] in ["inlink_context"]:
            addOrBuildBOWToIndexExcludingCurrent(fwriters[indexName], guid,  cp.Corpus.TEST_FILES, index_max_year, method, parameter)
        elif index_data["type"]=="ilc_mashup":
            bows=doc_representation.mashupBOWinlinkMethods(meta,[guid], index_max_year, indexNames[indexName], full_corpus=True)
            if not bows:
                print("ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter)
                continue
            addLoadedBOWsToIndex(fwriters[indexName], guid, bows, index_data)

def addOrBuildBOWToIndex(writer, guid, index_data, full_corpus=False):
    """
        Loads JSON file with BOW data to doc in index, NOT filtering for anything
    """
    bow_filename=cp.Corpus.cachedDataIDString("bow",guid,index_data)
    bows=cp.Corpus.loadCachedJson(bow_filename)
    if not bows:
        bows=prebuildMulti(index_data["method"],
                           index_data["parameter"],
                           index_data["function_name"],
                           None,
                           None,
                           guid,
                           False,
                           []) #!TODO rhetorical_annotations here?

    assert isinstance(bows,list)
    addLoadedBOWsToIndex(writer, guid, bows, index_data)

def addOrBuildBOWToIndexExcludingCurrent(writer, guid, exclude_list, max_year, index_data, full_corpus=False):
    """
        Loads JSON file with BOW data to index, filtering for
        inlink_context, excluding what bits
        came from the current exclude_list, posterior year, same author, etc.
    """
    bow_filename=cp.Corpus.cachedDataIDString("bow",guid,index_data)
    bows=cp.Corpus.loadCachedJson(bow_filename)
    if not bows:
        bows=prebuildMulti(index_data["method"],
                           index_data["parameter"],
                           index_data["function_name"],
                           None,
                           None,
                           guid,
                           False,
                           []) #!TODO rhetorical_annotations here?

    assert isinstance(bows,list)

    # joinTogetherContext?
    bows=doc_representation.filterInlinkContext(bows, exclude_list, max_year, full_corpus=full_corpus)

    assert isinstance(bows,list)
    addLoadedBOWsToIndex(writer, guid, bows)


def addLoadedBOWsToIndex(writer, guid, bows, bow_info):
    """
        Adds loaded bows as pointer to a file [fn]/guid [guid]

        :param writer: writer instance
        :param guid: ditto
        :param bows: list of dicts [{"title":"","abstract":""},{},...]
        :param bow_info: a dict with info about the bow being added, e.g. method that generated it and parameter
    """
    i=0
    base_metadata=cp.Corpus.getMetadataByGUID(guid)
    assert(base_metadata)

    assert isinstance(bows,list)
    for new_doc in bows: # takes care of passage
        if len(new_doc) == 0: # if the doc dict contains no fields
            continue

        fields_to_process=[field for field in new_doc if field not in doc_representation.FIELDS_TO_IGNORE]

        if len(fields_to_process) == 0: # if there is no overlap in fields to add
            continue

        numTerms={}
        total_numTerms=0

        metadata=deepcopy(base_metadata)

        for field in fields_to_process:
            field_len=len(new_doc[field].split())   # total number of terms
##            unique_terms=len(set(new_doc[field].split()))  # unique terms
            numTerms[field]=field_len
            # ignore fields that start with _ for total word count
            if field[0] != "_":
                total_numTerms+=field_len

        bow_info["passage_num"]=i
        bow_info["total_passages"]=len(bows)
        bow_info["total_numterms"]=total_numTerms

        ADD_DOCUMENT_FUNCTION(writer, new_doc, metadata, fields_to_process, bow_info)
        i+=1


def main():
    pass

if __name__ == '__main__':
    main()
