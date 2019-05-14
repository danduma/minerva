# classes that encapsulate all Lucene retrieval
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import re,json,sys
from string import punctuation

import lucene
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.document import Field
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import FieldAgnosticQueryParser
from org.apache.lucene.search.similarities import DefaultSimilarity, FieldAgnosticSimilarity
from org.apache.lucene.util import Version as LuceneVersion
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import TopScoreDocCollector
from java.io import File

from collections import namedtuple

from .base_retrieval import BaseRetrieval, SPECIAL_FIELDS_FOR_TESTS, MAX_RESULTS_RECALL
from .stored_formula import StoredFormula
from six.moves import range

class LuceneRetrieval(BaseRetrieval):
    """
        Encapsulates the Lucene retrieval engine
    """
    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
        self.index_path=index_path
        directory = SimpleFSDirectory(File(self.index_path))
        self.analyzer = StandardAnalyzer(LuceneVersion.LUCENE_CURRENT)
        self.reader=DirectoryReader.open(directory)
        self.searcher = IndexSearcher(self.reader)

        # uncomment one of these lines to change the type of parser, query and weight used
        if use_default_similarity:
            self.query_parser=QueryParser
        else:
            self.query_parser=FieldAgnosticQueryParser

        if use_default_similarity:
            similarity=DefaultSimilarity()
            self.useExplainQuery=False
        else:
            similarity=FieldAgnosticSimilarity()
            self.useExplainQuery=True
        # by default, FieldAgnosticSimilarity uses coord factor, can be disabled
##        similarity.useCoord=False

        self.searcher.setSimilarity(similarity)
        self.method=method # never used?
        self.logger=logger

    def runQueryViaExplain(self,query, max_results):
        """
            Really crappy solution to make sure that explanations and searches are the same
            while I fix Lucene
        """
        results=[]

        index=0
        for index in range(self.reader.numDocs()):
            explanation=self.searcher.explain(query,index)
            score=explanation.getValue()
##            match=re.search(r"(.*?)\s=",explanation.toString(),re.IGNORECASE|re.DOTALL)
##            if match:
##                score=float(match.group(1))
            hit=namedtuple("Hit",["doc","score"])
            hit.doc=index
            hit.score=score
##            heapq.heappush(results,hit)
            results.append(hit)

        results.sort(key=lambda x:x.score,reverse=True)

        if max_results < self.reader.numDocs():
            results=results[:max_results]

        return results

    def runQuery(self, structured_query, max_results=MAX_RESULTS_RECALL):
        """
            LOTS OF SWEET LUCENE
        """
        original_query=structured_query

        if not structured_query or len(structured_query) == 0 :
            return []

        self.last_query=structured_query
        query_text=self.rewriteQuery(structured_query["structured_query"], ["text"])

        try:
            query = self.query_parser(lucene.Version.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
        except:
            print("Lucene exception:",sys.exc_info()[:2])
            return None

        structured_query["lucene_query"]=query_text

        if self.useExplainQuery:
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,max_results)
        else:
            collector=TopScoreDocCollector.create(max_results, True)
            self.searcher.search(query, collector)
            hits = collector.topDocs().scoreDocs

##        print("Found %d document(s) that matched query '%s':" % (hits.totalHits, query))
        res=[]


##        if len(hits.scoreDocs) ==0:
##            print "Original query:",original_query
##            print "Query:", query

        for hit in hits:
            doc = self.searcher.doc(hit.doc)
            metadata= json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))
        return res


    def formulaFromExplanation(self, query, doc_id):
        """
            Runs .explain() for one query/doc pair, generates and returns a \
            StoredFormula instance from it

            :param query: Elastic DSL Query
            :param doc_id: id of document to run .explain() for
            :returns:
        """
        explanation=self.searcher.explain(query,doc_id)

        formula=StoredFormula()
        formula.fromLuceneExplanation(explanation)
        return formula


class LuceneRetrievalBoost(LuceneRetrieval):
    """
        Use Lucene for retrieval boosting different AZ fields differently
    """

    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
        LuceneRetrieval.__init__(self, index_path, method, logger, use_default_similarity)

    def runQuery(self, structured_query, parameters, test_guid, max_results=MAX_RESULTS_RECALL):
        """
            Run the query, return a list of tuples (score,metadata) of top docs
        """

        if not structured_query or len(structured_query) == 0 :
            return []

        self.last_query=structured_query

        query_text=self.rewriteQuery(structured_query["structured_query"],parameters,test_guid)

        try:
            query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
        except lucene.JavaError:
            # this is to deal with the "too many boolean clauses" exception
            print("Lucene exception:",sys.exc_info()[:2])
            print("Query:",query_text)
            return []

        structured_query["lucene_query"]=query_text

        if self.useExplainQuery:
            # TODO remove this completely, use DisjunctionMax
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,max_results)
        else:
            collector=TopScoreDocCollector.create(max_results, True)
            self.searcher.search(query, collector)
            hits = collector.topDocs().scoreDocs
        res=[]

        # explain the query
        if self.logger:
            self.logger.logReport(query_text+"\n")

            if self.logger.full_citation_id in self.logger.citations_extra_info:
                max_explanations=len(hits)
            else:
                max_explanations=1

            for index in range(max_explanations):
                self.logger.logReport(self.searcher.explain(query,index))

        for hit in hits:
            doc = self.searcher.doc(hit.doc)
            metadata=json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print(query_text,"\n", hits, "\n", res, "\n")

        del hits
        del query
        del query_text

        return res

##class precomputedExplainRetrieval(LuceneRetrievalBoost):
##    """
##        Class that runs the explain pipeline and extracts the data from a lucene
##        explanation to then be able to run the same retrieval with different
##        field boosts.
##
##        All it does is this one-time retrieval + explanation. The further testing
##        of parameters is done in measurePrecomputedResolution()
##    """
##
##    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
##        LuceneRetrievalBoost.__init__(self, index_path, method, logger, use_default_similarity)
##
##    def precomputeExplain(self, query, parameters, test_guid, max_results=MAX_RESULTS_RECALL):
##        """
##            Runs Lucene retrieval, extracts and stores the explanations in a dict:
##                one entry per potential paper, fields to set weights to as sub-entries
##        """
##
##        query_text=self.generateLuceneQuery(query, parameters, test_guid)
##        try:
##            query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
##        except:
##            print("Lucene exception:",sys.exc_info()[:2])
##            return None
##
##        results=[]
##
##        index=0
##
##        # we first run a search, so that we only need to run the explain pipeline
##        # for doc that actually match the query. This is just to speed up
##        # whole-collection precomputing of retrieval formulas
##        collector=TopScoreDocCollector.create(max_results, True)
##        self.searcher.search(query, collector)
##        hits = collector.topDocs().scoreDocs
##        doc_list=[hit.doc for hit in hits]
##
##        for index in doc_list:
##            explanation=self.searcher.explain(query,index)
##
##            formula=StoredFormula()
##            formula.fromExplanation(explanation)
##
##            if formula.formula["coord"] != 0:
##                doc = self.searcher.doc(index)
##                results.append({"index":index,"guid":doc.get("guid"),"formula":formula.formula})
##        return results


def main():
    pass

if __name__ == '__main__':
    main()
