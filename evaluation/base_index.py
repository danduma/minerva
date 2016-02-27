# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import sys, json, datetime, math
from copy import deepcopy
##from tqdm import tqdm
from progressbar import ProgressBar, SimpleProgress, Bar, ETA

import minerva.db.corpora as cp
import minerva.proc.doc_representation as doc_representation
from minerva.proc.general_utils import loadFileText, writeFileText, ensureDirExists

ES_TYPE_DOC="doc"

class BaseIndexer(object):
    """
        Prebuilds BOWs etc. for tests
    """
    def __init__(self):
        """
        """
        pass

    def buildIndexes(self, testfiles, methods):
        """
            For every test file in [testfiles],
                create index
                for every in-collection reference,
                    add all of the BOWs of methods in [methods] to index
        """
        self.initializeIndexer()

##        print("Prebuilding Lucene indeces in ",baseFullIndexDir)

        count=0
        for fn in testfiles:
            count+=1
            print("Building Lucene index for file",count,"/",len(testfiles),":",fn)
            test_guid=cp.Corpus.getFileUID(fn)

            fwriters={}
            baseFileIndexDir=baseFullIndexDir+test_guid+os.sep
            ensureDirExists(baseFileIndexDir)

            doc=cp.Corpus.loadSciDoc(test_guid)
            if not doc:
                print("Error loading XML for", test_guid)
                continue

            indexNames=doc_representation.getDictOfLuceneIndeces(methods)

            for indexName in indexNames:
##                actual_dir=cp.Corpus.paths.fileLuceneIndex+os.sep+test_guid+os.sep+indexName
                actual_dir=cp.Corpus.getRetrievalIndexPath(test_guid, indexName, full_corpus=False)
                fwriters[indexName]=createIndexWriter(actual_dir)

            # old way, assuming the documents are fine and one can just load all in-collection references
            # ...NOT! must select them using the same method that gets the resolvable CITATIONS
            # updated! Should work well now
            for ref in doc["references"]:
                match=cp.Corpus.matchReferenceInIndex(ref)
                if match:

                    for indexName in indexNames:
                        # get the maximum year to create inlink_context descriptions from
                        if indexNames[indexName]["options"].get("max_year",False) == True:
                            max_year=cp.Corpus.getMetadataByGUID(test_guid)["year"]
                        else:
                            max_year=None

                        method=indexNames[indexName]["method"]
                        parameter=indexNames[indexName]["parameter"]
                        ilc_parameter=indexNames[indexName].get("ilc_parameter","")

                        if indexNames[indexName]["type"] in ["standard_multi"]:
                            addPrebuiltBOWtoIndex(fwriters[indexName], match["guid"], method, parameter)
                        elif indexNames[indexName]["type"] in ["inlink_context"]:
                            addPrebuiltBOWtoIndexExcludingCurrent(fwriters[indexName], match["guid"], [test_guid], max_year, method, parameter)
                        elif methods[method]["type"]=="ilc_mashup":

                            bows=doc_representation.mashupBOWinlinkMethods(match,[test_guid], max_year, indexNames[indexName])
                            if not bows:
                                print("ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter)
                                continue

                            addLoadedBOWsToIndex(fwriters[indexName], match["guid"], bows,
                            {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

            for fwriter in fwriters:
                fwriters[fwriter].close()

    def buildGeneralIndex(self, testfiles, methods):
        """
            Creates one Lucene index for each method and parameter, adding all files to each
        """

##        print ("Prebuilding Lucene indeces in ",baseFullIndexDir)

        print ("Building global index...")
        fwriters={}

        indexNames=doc_representation.getDictOfLuceneIndeces(methods)
        for indexName in indexNames:
            actual_dir=cp.Corpus.getRetrievalIndexPath("ALL_GUIDS", indexName, full_corpus=True)
##            cp.Corpus.paths.get("fullLuceneIndex","")+indexName
            fwriters[indexName]=self.createIndexWriter(actual_dir)

        ALL_GUIDS=cp.Corpus.listPapers()
    ##    ALL_GUIDS=["j98-2002"]

        dot_every_xfiles=max(len(ALL_GUIDS) / 1000,1)
        print("Adding",len(ALL_GUIDS),"files:")

        now1=datetime.datetime.now()
        count=0
##        for guid in tqdm(ALL_GUIDS, desc="Adding file"):
        widgets = ['Adding file: ', SimpleProgress(), ' ', Bar(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=100).start()
        for guid in ALL_GUIDS:
            count+=1

            progress.update(count)
##            print("Adding file:",count,"/",len(ALL_GUIDS),":",guid)

            meta=cp.Corpus.getMetadataByGUID(guid)
            if not meta:
                print("Error: can't load metadata for paper",guid)
                continue

            for indexName in indexNames:
                # get the maximum year to create inlink_context descriptions from
                if indexNames[indexName]["options"].get("max_year",False) == True:
                    max_year=cp.Corpus.getMetadataByGUID(test_guid)["year"]
                else:
                    max_year=None

                method=indexNames[indexName]["method"]
                parameter=indexNames[indexName]["parameter"]
                ilc_parameter=indexNames[indexName].get("ilc_parameter","")

                if indexNames[indexName]["type"] in ["standard_multi"]:
                    self.addPrebuiltBOWtoIndex(fwriters[indexName], guid, method, parameter)
                elif indexNames[indexName]["type"] in ["inlink_context"]:
                    self.addPrebuiltBOWtoIndexExcludingCurrent(fwriters[indexName], guid,  testfiles, max_year, method, parameter)
                elif methods[method]["type"]=="ilc_mashup":
                    bows=doc_representation.mashupBOWinlinkMethods(match,[test_guid], max_year, indexNames[indexName], full_corpus=True)
                    if not bows:
                        print("ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter)
                        continue

                    self.addLoadedBOWsToIndex(fwriters[indexName], match["guid"], bows,
                    {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

    ##        for indexName in indexNames:
    ##            method=indexNames[indexName]["method"]
    ##            parameter=indexNames[indexName]["parameter"]
    ##            ilc_parameter=indexNames[indexName].get("ilc_parameter","")
    ##
    ##            if indexNames[indexName]["type"] in ["standard_multi","inlink_context"]:
    ##                addPrebuiltBOWtoIndexExcludingCurrent(fwriters[indexName], guid, [], method, parameter)
    ##            elif methods[method]["type"]=="ilc_mashup":
    ##                bows=mashupBOWinlinkMethods(meta,["NONE"],indexNames[indexName]["mashup_method"], parameter,ilc_parameter)
    ##                if not bows:
    ##                    print "ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter
    ##                    continue
    ##                addLoadedBOWsToIndex(fwriters[indexName], guid, bows,
    ##                {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

        for fwriter in fwriters:
            fwriters[fwriter].close()

    def addPrebuiltBOWtoIndex(self, writer, guid, method, parameter, full_corpus=False):
        """
            Loads JSON file with BOW data to Lucene doc in index, NOT filtering for anything
        """
        method_dict={"method":method,"parameter":parameter}
        bow_filename=cp.Corpus.cachedDataIDString("bow",guid,method_dict)
        bows=cp.Corpus.loadCachedJson(bow_filename)

        assert isinstance(bows,list)
        self.addLoadedBOWsToIndex(writer, guid, bows, method_dict)

    def addPrebuiltBOWtoIndexExcludingCurrent(self, writer, guid, exclude_list, max_year, method, parameter, full_corpus=False):
        """
            Loads JSON file with BOW data to Lucene index, filtering for
            inlink_context, excluding what bits
            came from the current exclude_list, posterior year, same author, etc.
        """
        method_dict={"method":method,"parameter":parameter}
        bow_filename=cp.Corpus.cachedDataIDString("bow",guid,method_dict)
        bows=cp.Corpus.loadCachedJson(bow_filename)

        assert isinstance(bows,list)

        # joinTogetherContext?
        bows=doc_representation.filterInlinkContext(bows, exclude_list, max_year, full_corpus=full_corpus)

        assert isinstance(bows,list)
        self.addLoadedBOWsToIndex(writer, guid, bows)


    def addLoadedBOWsToIndex(self, writer, guid, bows, bow_info):
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

            self.addDocument(writer, new_doc, metadata, fields_to_process, bow_info)
            i+=1



#-------------------------------------------------------------------------------
#  Methods to be overriden in descendant classes
#-------------------------------------------------------------------------------

    def initializeIndexer(self):
        """
            Any previous step that is needed before indexing documents
        """
        pass

    def createIndexWriter(self, actual_dir, max_field_length=20000000):
        """
            Returns an IndexWriter object created for the actual_dir specified
        """
        raise NotImplementedError

    def addDocument(self, writer, new_doc, metadata, fields_to_process, bow_info):
        """
            Add a document to the index. To be overriden by descendant classes.

            :param new_doc: dict of fields with values
            :type new_doc:dict
            :param metadata: ditto
            :type metadata:dict
            :param fields_to_process: only add these fields from the doc dict
            :type fields_to_process:list
            :param bow_info: a dict with info on the bow: how it was generated, etc.
        """
        raise NotImplementedError



def main():
    pass

if __name__ == '__main__':
    main()
