# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import sys, json, datetime, math
from copy import deepcopy
import logging
##from tqdm import tqdm
##from progressbar import ProgressBar, SimpleProgress, Bar, ETA
from minerva.proc.results_logging import ProgressIndicator

import minerva.db.corpora as cp
import minerva.evaluation.base_pipeline as doc_representation
from minerva.evaluation.base_pipeline import getDictOfTestingMethods
from minerva.proc.doc_representation import getDictOfLuceneIndeces
from minerva.proc.general_utils import loadFileText, writeFileText, ensureDirExists
from index_functions import addBOWsToIndex

from minerva.proc.nlp_functions import CORESC_LIST

from minerva.squad.tasks import addToindexTask

ES_TYPE_DOC="doc"

class BaseIndexer(object):
    """
        Prebuilds BOWs etc. for tests
    """
    def __init__(self, use_celery=False):
        """
        """
        self.use_celery=use_celery

    def buildIndexes(self, testfiles, methods):
        """
            For every test file in [testfiles],
                create index
                for every in-collection reference,
                    add all of the BOWs of methods in [methods] to index
        """
        self.initializeIndexer()

        count=0
        for guid in testfiles:
            count+=1
            print("Building index: paper ",count,"/",len(testfiles),":",guid)

            fwriters={}
            doc=cp.Corpus.loadSciDoc(guid)
            if not doc:
                print("Error loading SciDoc for", guid)
                continue

            indexNames=getDictOfLuceneIndeces(methods)

            for indexName in indexNames:
                actual_dir=cp.Corpus.getRetrievalIndexPath(guid, indexName, full_corpus=False)
                fwriters[indexName]=self.createIndexWriter(actual_dir)

            # old way, assuming the documents are fine and one can just load all in-collection references
            # ...NOT! must select them using the same method that gets the resolvable CITATIONS
            # updated! Should work well now
##            for ref in doc["references"]:
##                match=cp.Corpus.matcher.matchReference(ref)
##                if match:
##                    ref_guid=match["guid"]
            # even newer way: just use the precomputed metadata.outlinks
            outlinks=cp.Corpus.getMetadataByGUID(guid)["outlinks"]
            for ref_guid in outlinks:
                addBOWsToIndex(ref_guid,indexNames,9999,fwriters)
                # TODO integrate this block below into addBOWsToIndex
##                for indexName in indexNames:
##                    # get the maximum year to create inlink_context descriptions from
##                    if indexNames[indexName]["options"].get("max_year",False) == True:
##                        max_year=cp.Corpus.getMetadataByGUID(test_guid)["year"]
##                    else:
##                        max_year=None

            for fwriter in fwriters:
                fwriters[fwriter].close()

    def listFieldsToIndex(self, index_data):
        """
            Returns a list of fields to NOT store, only index.
        """
        # TODO make classes for the extractors, so that each class reports its fields
        # WARNING this function is super hackish for tests
        # like, seriously, I need to change this

        if index_data["type"] in ["annotated_boost", "ilc_annotated_boost"]:
            return index_data["runtime_parameters"]
        elif index_data["type"] in ["inlink_context"]:
            pass
        elif index_data["type"] in ["ilc_mashup"]:
            return CORESC_LIST + ["ilc_CSC_"+zone for zone in CORESC_LIST]
        elif index_data["type"] in ["standard_multi"]:
            if index_data["method"] in ["az_annotated", "ilc_annotated"]:
                return CORESC_LIST + ["ilc_CSC_"+zone for zone in CORESC_LIST]
            pass


    def buildGeneralIndex(self, exp, options):
        """
            Creates one index for each method and parameter, adding all files to each
        """
        print ("Building global index...")
        fwriters={}

        index_max_year=exp.get("index_max_year",None)

        indexNames=getDictOfLuceneIndeces(exp["prebuild_general_indexes"])
        for entry_name in indexNames:
            entry=indexNames[entry_name]
            entry["function_name"]=exp["prebuild_bows"][entry["bow_name"]]["function_name"]

        max_results=options.get("max_files_to_process",sys.maxint)

        ALL_GUIDS=cp.Corpus.listPapers("metadata.year:<=%d" % index_max_year,  max_results=max_results)
        for indexName in indexNames:
            actual_dir=cp.Corpus.getRetrievalIndexPath("ALL_GUIDS", indexName, full_corpus=True)
            fields=self.listFieldsToIndex(indexNames[indexName])
            self.createIndex(actual_dir,fields)
            fwriters[indexName]=self.createIndexWriter(actual_dir)

        print("Adding",len(ALL_GUIDS),"files:")

        if not self.use_celery:
##            widgets = ['Adding file: ', SimpleProgress(), ' ', Bar(), ' ', ETA()]
##            progress = ProgressBar(widgets=widgets, maxval=100).start()
            progress=ProgressIndicator(True, len(ALL_GUIDS), print_out=False)
            for guid in ALL_GUIDS:
                addBOWsToIndex(guid, indexNames, index_max_year, fwriters)
                progress.showProgressReport("Adding papers to index")
            for fwriter in fwriters:
                fwriters[fwriter].close()
        else:
            print("Queueing up files for import...")
            for guid in ALL_GUIDS:
                addToindexTask.apply_async(args=[
                                                guid,
                                                indexNames,
                                                index_max_year,
                                                ],
                                            queue="add_to_index")


#-------------------------------------------------------------------------------
#  Methods to be overriden in descendant classes
#-------------------------------------------------------------------------------

    def initializeIndexer(self):
        """
            Any previous step that is needed before indexing documents
        """
        pass

    def createIndex(self, index_name, fields):
        """
            Create the actual index. Elastic requires this in order to
            specify the mapping, Lucene doesn't
        """
        raise NotImplementedError

    def createIndexWriter(self, actual_dir, max_field_length=20000000):
        """
            Returns an IndexWriter object created for the actual_dir specified
        """
        raise NotImplementedError

##    def addDocument(self, writer, new_doc, metadata, fields_to_process, bow_info):
##        """
##            Add a document to the index. To be overriden by descendant classes.
##
##            :param new_doc: dict of fields with values
##            :type new_doc:dict
##            :param metadata: ditto
##            :type metadata:dict
##            :param fields_to_process: only add these fields from the doc dict
##            :type fields_to_process:list
##            :param bow_info: a dict with info on the bow: how it was generated, etc.
##        """
##        raise NotImplementedError


def main():
    pass

if __name__ == '__main__':
    main()
