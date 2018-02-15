# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import sys, json, datetime, math

import db.corpora as cp
import proc.doc_representation as doc_representation
from proc.general_utils import loadFileText, writeFileText, ensureDirExists

from .base_index import BaseIndexer

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


class LuceneIndexer(BaseIndexer):
    """
        Prebuilds BOWs etc. for tests
    """
    def __init__(self):
        """
        """
        pass

    def initializeIndexer(self):
        """
            Initializes the Java VM, creates directories if needed
        """
        print("Initializing VM...")
        lucene.initVM(maxheap="768m")

        baseFullIndexDir=cp.Corpus.paths.fileLuceneIndex+os.sep
        ensureDirExists(baseFullIndexDir)

    def createIndexWriter(self, actual_dir, max_field_length=20000000):
        """
            Returns an IndexWriter object created for the actual_dir specified
        """
        ensureDirExists(actual_dir)
        index = SimpleFSDirectory(File(actual_dir))
        analyzer = StandardAnalyzer(LuceneVersion.LUCENE_CURRENT)

        writerConfig=IndexWriterConfig(LuceneVersion.LUCENE_CURRENT, analyzer)
        similarity=FieldAgnosticSimilarity()

        writerConfig.setSimilarity(similarity)
        writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    ##    res= IndexWriter(index, analyzer, True, IndexWriter.MaxFieldLength(max_field_length))
        res= IndexWriter(index, writerConfig)
        res.deleteAll()
        return res

    def addDocument(self, writer, new_doc, metadata, fields_to_process, bow_info):
        """
            Add a document to the index. Does this using direct Lucene access.

            :param new_doc: dict of fields with values
            :type new_doc:dict
            :param metadata: ditto
            :type metadata:dict
            :param fields_to_process: only add these fields from the doc dict
            :type fields_to_process:list
        """
        doc = Document()
        total_numTerms=bow_info["total_numterms"]
        # each BOW now comes with its field
        for field in fields_to_process:
            field_object=Field(field, new_doc[field], Field.Store.NO, Field.Index.ANALYZED, Field.TermVector.YES)
##            boost=math.sqrt(numTerms[field]) / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
            boost=1 / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
            field_object.setBoost(float(boost))
            doc.add(field_object)

        json_metadata=json.dumps(metadata)
        doc.add(Field("guid", guid, Field.Store.YES, Field.Index.ANALYZED))
        doc.add(Field("bow_info", json.dumps(bow_info), Field.Store.YES, Field.Index.NO))
        doc.add(Field("metadata", json_metadata, Field.Store.YES, Field.Index.NO))
        doc.add(Field("year_from", metadata["year"], Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(doc)

def main():

    pass

if __name__ == '__main__':
    main()
