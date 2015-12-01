# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from minerva.util.general_utils import loadFileText, writeFileText

import minerva.util.context_extract as context_extract
from corpora import Corpus

import sys, json, datetime, math

# lucene
import lucene

from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.document import Field
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import DefaultSimilarity, FieldAgnosticSimilarity
from org.apache.lucene.util import Version as LuceneVersion
from org.apache.lucene.index import IndexWriterConfig
from java.io import File


# this is the corpus instance used for all experiments
print "Loading corpus..."
drive="g"
Corpus.connectCorpus(drive+r":\NLP\PhD\aac")
# this is to exclude the papers that Victor said
EXCLUDE_MORE_RECENT=False


def prebuildBOWsForTests(parameters, maxfiles=1000000, FILE_LIST=None, force_prebuild=False):
    """
        Generates BOWs for each document from its inlinks, stores them in .pic file

        It would be nice to be able to store everything so one doesn't have to generate every
        BOW / VSM representation every single run

    """

    def prebuildMulti(name,parameters,function, guid, doc, doctext, force_prebuild):
        if not force_prebuild:
            params=corpora.Corpus.selectBOWParametersToPrebuild(guid,name,parameters)
        else:
            params=parameters

        if len(params) > 0:
            # changed this so doc only gets loaded if absolutely necessary
            if not doc:
                doc=corpora.Corpus.loadSciDoc(guid)
                corpora.Corpus.annotateDoc(doc)
                doctext=doc.getFullDocumentText()

            all_bows=function(doc,parameters=params, doctext=doctext)
            print "Saving prebuilt "+name+" BOW for",doc["metadata"]["guid"]

            for param in params:
##                if guid=="a00-2027":
##                    print doc["metadata"]["guid"], param
                corpora.Corpus.savePrebuiltBOW(doc["metadata"]["guid"],name, param, all_bows[param])
        return [doc,doctext]

    count=0
    if FILE_LIST:
        corpora.Corpus.ALL_FILES=FILE_LIST
    else:
        corpora.Corpus.ALL_FILES=corpora.Corpus.listAllPapers()

    print "Loading AZ/CFC classifiers"
    corpora.Corpus.loadAnnotators()

    print "Prebuilding BOWs for", min(len(corpora.Corpus.ALL_FILES),maxfiles), "files..."
    numfiles=min(len(corpora.Corpus.ALL_FILES),maxfiles)

    for guid in corpora.Corpus.ALL_FILES[:maxfiles]:
        count+=1
        print "Processing ", count, "/", numfiles, " - ", guid

        doctext=None
        doc=None

        # prebuild BOWs for all entries in the build_db
        for entry in parameters:
            doc,doctext=prebuildMulti(entry,parameters[entry]["parameters"],parameters[entry]["function"],guid, doc, doctext, force_prebuild)

def createLuceneIndexWriter(actual_dir, max_field_length=20000000):
    """
        Returns an IndexWriter object created for the actual_dir specified
    """
    ensureDirExists(actual_dir)
    index = SimpleFSDirectory(File(actual_dir))
##    analyzer = StandardAnalyzer(lucene.Version.LUCENE_CURRENT)
    analyzer = StandardAnalyzer(LuceneVersion.LUCENE_CURRENT)

##    writerConfig=IndexWriterConfig(lucene.Version.LUCENE_CURRENT, analyzer)
    writerConfig=IndexWriterConfig(LuceneVersion.LUCENE_CURRENT, analyzer)
    similarity=FieldAgnosticSimilarity()

    writerConfig.setSimilarity(similarity)
##    writerConfig.setOpenMode(lucene.IndexWriterConfig.OpenMode.CREATE)
    writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

##    res= IndexWriter(index, analyzer, True, IndexWriter.MaxFieldLength(max_field_length))
    res= IndexWriter(index, writerConfig)
    res.deleteAll()
    return res


def addLoadedBOWsToLuceneIndex(writer, guid, bows, metadata):
    """
        Adds loaded bows as pointer to a file [fn]/guid [guid]

        bows = list of dicts [{"title":"","abstract":""},{},...]
    """
    i=0
    assert isinstance(bows,list)
    for new_doc in bows: # takes care of passage too
        if len(new_doc) == 0: # if the doc dict contains no fields
            continue

        fields_to_process=[field for field in new_doc if field not in context_extract.FIELDS_TO_IGNORE]

        numTerms={}
        total_numTerms=0

        for field in fields_to_process:
            field_len=len(new_doc[field].split())   # total number of terms
##            unique_terms=len(set(new_doc[field].split()))  # unique terms
            numTerms[field]=field_len
            # ignore fields that start with _ for total word count
            if field[0] != "_":
                total_numTerms+=field_len

        doc = Document()
        # each BOW now comes with its field
        for field in fields_to_process:
            field_object=Field(field, new_doc[field], Field.Store.NO, Field.Index.ANALYZED, Field.TermVector.YES)
##            boost=math.sqrt(numTerms[field]) / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
            boost=1 / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
            field_object.setBoost(float(boost))
            doc.add(field_object)
##            print field,field_object.boost()

    ##        metadata["filename"]=fn
        metadata["guid"]=guid
        metadata["passage_num"]=i
        metadata["total_passages"]=len(bows)
        json_metadata=json.dumps(metadata)

##        doc.add(Field("filename", fn, Field.Store.YES, Field.Index.NO))
        doc.add(Field("guid", guid, Field.Store.YES, Field.Index.NO))
##        doc.add(Field("passage_num", i, Field.Store.YES, Field.Index.NO))
##        doc.add(Field("total_passages", len(bows), Field.Store.YES, Field.Index.NO))
        doc.add(Field("metadata", json_metadata, Field.Store.YES, Field.Index.NO))
        doc.add(Field("year_from", guid, Field.Store.YES, Field.Index.NO))
        writer.addDocument(doc)
        i+=1


def addPrebuiltBOWtoLuceneIndex(writer, guid, method, parameter, full_corpus=False):
    """
        Loads JSON file with BOW data to Lucene doc in index, NOT filtering for anything
    """
    bow_filename=corpora.Corpus.prebuiltFilePath(guid,method,parameter)
    bows=json.loads(loadFileText(bow_filename))

    assert isinstance(bows,list)
    addLoadedBOWsToLuceneIndex(writer, guid, bows, {"method":method,"parameter":parameter})


def addPrebuiltBOWtoLuceneIndexExcludingCurrent(writer, guid, exclude_list, max_year, method, parameter, full_corpus=False):
    """
        Loads JSON file with BOW data to Lucene index, filtering for
        inlink_context, excluding what bits
        came from the current exclude_list, posterior year, same author, etc.
    """

    bow_filename=corpora.Corpus.prebuiltFilePath(guid,method,parameter)
    bows=json.loads(loadFileText(bow_filename))
    assert isinstance(bows,list)

    # joinTogetherContext?
    bows=context_extract.filterInlinkContext(bows, exclude_list, max_year, full_corpus=full_corpus)

    assert isinstance(bows,list)
    addLoadedBOWsToLuceneIndex(writer, guid, bows, {"method":method,"parameter":parameter})


def prebuildLuceneIndexes(testfiles, methods):
    """
        For every test file in [testfiles],
            create index
            for every in-collection reference,
                add all of the BOWs of methods in [methods] to index
    """
    print "Initializing VM..."
    lucene.initVM(maxheap="768m")

    baseFullIndexDir=corpora.Corpus.dir_fileLuceneIndex+os.sep
    ensureDirExists(baseFullIndexDir)

    print "Prebuilding Lucene indeces in",baseFullIndexDir

    count=0
    for fn in testfiles:
        count+=1
        print "Building Lucene index for file",count,"/",len(testfiles),":",fn
        test_guid=corpora.Corpus.getFileUID(fn)

        fwriters={}
        baseFileIndexDir=baseFullIndexDir+test_guid+os.sep
        ensureDirExists(baseFileIndexDir)

        doc=corpora.Corpus.loadSciDoc(test_guid)
        if not doc:
            print "Error loading XML for", test_guid
            continue

        indexNames=context_extract.getDictOfLuceneIndeces(methods)

        for indexName in indexNames:
            actual_dir=corpora.Corpus.dir_fileLuceneIndex+os.sep+test_guid+os.sep+indexName
            fwriters[indexName]=createLuceneIndexWriter(actual_dir)

        # new way: either loads or generates resolvable citations
        matchable_references=[corpora.Corpus.getMetadataByGUID(x) for x in
         corpora.Corpus.getResolvableCitationsCache(fn,doc)["outlinks"]]

        # old way, assuming the documents are fine and one can just load all in-collection references
        # ...NOT! must select them using the same method that gets the resolvable CITATIONS
        # updated! Should work well now
        for ref in doc["references"]:
            match=corpora.Corpus.matchReferenceInIndex(ref)
            if match:
##        for match in matchable_references:

                for indexName in indexNames:
                    # get the maximum year to create inlink_context descriptions from
                    if indexNames[indexName]["options"].get("max_year",False) == True:
                        max_year=corpora.Corpus.getMetadataByGUID(test_guid)["year"]
                    else:
                        max_year=None

                    method=indexNames[indexName]["method"]
                    parameter=indexNames[indexName]["parameter"]
                    ilc_parameter=indexNames[indexName].get("ilc_parameter","")

                    if indexNames[indexName]["type"] in ["standard_multi"]:
                        addPrebuiltBOWtoLuceneIndex(fwriters[indexName], match["guid"], method, parameter)
                    elif indexNames[indexName]["type"] in ["inlink_context"]:
                        addPrebuiltBOWtoLuceneIndexExcludingCurrent(fwriters[indexName], match["guid"], [test_guid], max_year, method, parameter)
                    elif methods[method]["type"]=="ilc_mashup":

                        bows=context_extract.mashupBOWinlinkMethods(match,[test_guid], max_year, indexNames[indexName])
                        if not bows:
                            print "ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter
                            continue

                        addLoadedBOWsToLuceneIndex(fwriters[indexName], match["guid"], bows,
                        {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

        for fwriter in fwriters:
##            fwriters[fwriter].optimize()  # this is so 3.6.2
            fwriters[fwriter].close()

def buildGeneralLuceneIndex(testfiles,methods):
    """
        Creates one Lucene index for each method and parameter, adding all files to each
    """
    print "Initializing VM..."
    lucene.initVM(maxheap="768m")

    baseFullIndexDir=corpora.Corpus.dir_fullLuceneIndex
    ensureDirExists(baseFullIndexDir)

    print "Prebuilding Lucene indeces in",baseFullIndexDir

    print "Building global Lucene index..."
    fwriters={}

    indexNames=context_extract.getDictOfLuceneIndeces(methods)
    for indexName in indexNames:
        actual_dir=corpora.Corpus.dir_fullLuceneIndex+indexName
        fwriters[indexName]=createLuceneIndexWriter(actual_dir)

    ALL_GUIDS=corpora.Corpus.listAllPapers()
##    ALL_GUIDS=["j98-2002"]

    dot_every_xfiles=max(len(ALL_GUIDS) / 1000,1)
    print "Adding",len(ALL_GUIDS),"files:"

    now1=datetime.datetime.now()
    count=0
    for guid in ALL_GUIDS:
        count+=1

        print "Adding file",count,"/",len(testfiles),":",guid

        meta=corpora.Corpus.getMetadataByGUID(guid)
        if not meta:
            print "Error: can't load metadata for paper",guid
            continue

        for indexName in indexNames:
            # get the maximum year to create inlink_context descriptions from
            if indexNames[indexName]["options"].get("max_year",False) == True:
                max_year=corpora.Corpus.getMetadataByGUID(test_guid)["year"]
            else:
                max_year=None

            method=indexNames[indexName]["method"]
            parameter=indexNames[indexName]["parameter"]
            ilc_parameter=indexNames[indexName].get("ilc_parameter","")

            if indexNames[indexName]["type"] in ["standard_multi"]:
                addPrebuiltBOWtoLuceneIndex(fwriters[indexName], guid, method, parameter)
            elif indexNames[indexName]["type"] in ["inlink_context"]:
                addPrebuiltBOWtoLuceneIndexExcludingCurrent(fwriters[indexName], guid,  testfiles, max_year, method, parameter)
            elif methods[method]["type"]=="ilc_mashup":
                bows=context_extract.mashupBOWinlinkMethods(match,[test_guid], max_year, indexNames[indexName], full_corpus=True)
                if not bows:
                    print "ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter
                    continue

                addLoadedBOWsToLuceneIndex(fwriters[indexName], match["guid"], bows,
                {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

##        for indexName in indexNames:
##            method=indexNames[indexName]["method"]
##            parameter=indexNames[indexName]["parameter"]
##            ilc_parameter=indexNames[indexName].get("ilc_parameter","")
##
##            if indexNames[indexName]["type"] in ["standard_multi","inlink_context"]:
##                addPrebuiltBOWtoLuceneIndexExcludingCurrent(fwriters[indexName], guid, [], method, parameter)
##            elif methods[method]["type"]=="ilc_mashup":
##                bows=mashupBOWinlinkMethods(meta,["NONE"],indexNames[indexName]["mashup_method"], parameter,ilc_parameter)
##                if not bows:
##                    print "ERROR: Couldn't load prebuilt BOWs for mashup with inlink_context and ", method, ", parameters:",parameter, ilc_parameter
##                    continue
##                addLoadedBOWsToLuceneIndex(fwriters[indexName], guid, bows,
##                {"method":method,"parameter":parameter, "ilc_parameter":ilc_parameter})

    for fwriter in fwriters:
        fwriters[fwriter].close()

def main():
##   This function should be able to prebuild everything or just one of the steps
##   This file assumes a prebuilt corpus index that is loaded on corpora.Corpus.
##    the bob corpus index should be built in bobcorpus_ingest
##    1. preload/pickle SciXML
##    2. preload/pickle matchable citations and outlinks
##    3. prebuild BOWs for each file
##    4. prebuild lucene index for each method for each TEST file

    # superset of what I used for ACL14
##    methods={
##    "full_text":{"type":"standard_multi", "parameters":[1]},
##    "passage":{"type":"standard_multi", "parameters":[150,175,200,250,300,350,400,450]},
##    "title_abstract":{"type":"standard_multi", "parameters":[1]},
##    "inlink_context":{"type":"standard_multi", "parameters":[3, 5, 10, 15, 20, 30, 40, 50] },
##    "ilc_title_abstract":{"type":"ilc_mashup", "mashup_method":"title_abstract", "ilc_parameters":[10,20,30], "parameters":[1]},
##    "ilc_full_text":{"type":"ilc_mashup", "mashup_method":"full_text", "ilc_parameters":[10,20,30], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "mashup_method":"passage","ilc_parameters":[10, 20, 30], "parameters":[250,300,350]}
##    }

    # including AZ
    prebuild_bows={
    "full_text":{"function":context_extract.getDocBOWfull, "parameters":[1]},
    "title_abstract":{"function":context_extract.getDocBOWTitleAbstract, "parameters":[1]},
    "passage":{"function":context_extract.getDocBOWpassagesMulti, "parameters":[150,175,200,250,300,350,400,450]},
    "inlink_context":{"function":context_extract.generateDocBOWInlinkContext, "parameters":[5, 10, 15, 20, 30, 40, 50] },
    "ilc_AZ":{"function":context_extract.generateDocBOW_ILC_Annotated, "parameters":["paragraph","1up_1down","1up","1only"] },
    "az_annotated":{"function":context_extract.getDocBOWannotated, "parameters":[1]},
    "section_annotated":{"function":context_extract.getDocBOWannotatedSections, "parameters":[1]},
    }

    # bow_name is just about the name of the file containing the BOWs
    prebuild_indexes={
##    "full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
##    "title_abstract":{"type":"standard_multi", "bow_name":"title_abstract", "parameters":[1]},
##    "passage":{"type":"standard_multi", "bow_name":"passage", "parameters":[150,175,200,250,300,350,400,450]},
##    "inlink_context":{"type":"inlink_context", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50]},
##    "inlink_context_year":{"type":"inlink_context", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50], "options":{"max_year":True}},
##    "az_annotated":{"type":"standard_multi", "bow_methods":[("az_annotated",[1])], "parameters":[1]},
##    "section_annotated":{"type":"standard_multi", "bow_methods":[("section_annotated",[1])], "parameters":[1]},

##    # this is just ilc but split by AZ
    "ilc_AZ":{"type":"inlink_context", "bow_name":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"]},

##    "ilc_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_year_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1], "options":{"max_year":True}},
##    "ilc_section_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"section_annotated", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"passage","ilc_parameters":[5, 10, 20, 30, 40, 50], "parameters":[250,300,350]},

    # this is just normal az_annotated + normal ilc
##    "ilc_az_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context",  "mashup_method":"az_annotated", "ilc_parameters":[5, 10,20,30, 40, 50], "parameters":[1]},

##    # this is az-annotated text + az-annotated ilc
##    "az_ilc_az_":{"type":"ilc_mashup", "ilc_method":"ilc_AZ", "mashup_method":"az_annotated", "ilc_parameters":["paragraph","1up_1down","1up","1only"], "parameters":[1]},
    }

    prebuild_general_indexes={
    "full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
    }

##
##    d=context_extract.getDictOfLuceneIndeces(prebuild_indexes)
##    print json.dumps(d,indent=3)
##    print json.dumps(d.keys(), indent=3)

##    d=context_extract.getDictOfTestingMethods(testing_methods)
##    print json.dumps(d,indent=3)
##    print json.dumps(d.keys(), indent=3)

    FILE_LIST=["a00-2031", "h94-1020", "a00-2034", "w98-1106", "p93-1024", "p99-1001", "w99-0632", "p98-1013", "j03-4003", "j93-2004", "p99-1014", "p96-1008"]
##    prebuildBOWsForTests(prebuild_bows, FILE_LIST=FILE_LIST)

    # uncomment this to update all outlink metadata
##    corpora.Corpus.updateAllPapersMetadataOutlinks()

##    corpora.Corpus.TEST_FILES=loadFileList(corpora.Corpus.dir_fileDB_db+"ACL14_test_docs.txt")

    corpora.Corpus.TEST_FILES=corpora.Corpus.listPapers("num_in_collection_references >= 8")
    # do just x files for testing
##    corpora.Corpus.TEST_FILES=corpora.Corpus.TEST_FILES[:10]
    # this is to build only the necessary BOWs for the test files
    prebuild_list=corpora.Corpus.listIncollectionReferencesOfList(corpora.Corpus.TEST_FILES)
    # this will build BOWs for ALL files for the general index
##    prebuild_list=corpora.Corpus.listAllPapers()

    print "Prebuilding BOWs for",len(prebuild_list),"files"

##    prebuildBOWsForTests(prebuild_bows,FILE_LIST=prebuild_list)
    prebuildLuceneIndexes(corpora.Corpus.TEST_FILES, prebuild_indexes)

##    prebuildLuceneIndexes(["j02-3001"], prebuild_indexes)
##    buildGeneralLuceneIndex(corpora.Corpus.TEST_FILES,prebuild_general_indexes)

    pass

if __name__ == '__main__':
    main()
