#-------------------------------------------------------------------------------
# Name:        lucene_retrieval
# Purpose:      classes that encapsulate all Lucene retrieval
#
# Author:      dd
#
# Created:     02/04/2015
# Copyright:   (c) dd 2015
#-------------------------------------------------------------------------------

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
from org.apache.lucene.search import TopScoreDocCollector;
from java.io import File

from collections import namedtuple

SPECIAL_FIELDS_FOR_TESTS=["inlink_context"]
MAX_RESULTS_RECALL=200

import context_extract
import re,json,sys

class luceneRetrieval:
    """
        Encapsulates the Lucene retrieval engine
    """
    def __init__(self, indexDir, method, logger=None, use_default_similarity=False):
        self.indexDir=indexDir
        directory = SimpleFSDirectory(File(self.indexDir))
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
        self.max_results=MAX_RESULTS_RECALL
        self.method=method # never used?
        self.logger=logger

    def cleanupQuery(self,query):
        """
            Remove symbols from the query that can lead to errors when parsing the query
        """
        rep_list=["~", "and", "^","\"","+","-","(",")", "{","}","[","]","?",":","*"]
        rep_list.extend(context_extract.punctuation)
        query=query.lower()
        for r in rep_list:
            query=query.replace(r," ")
        query=re.sub(r"\s+"," ",query)
        query=query.strip()
        return query

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

    def runQuery(self, query):
        """
            LOTS OF SWEET LUCENE
        """
        original_query=query
        query=self.cleanupQuery(query)
        if query=="":
            return None

        self.last_query=query

        try:
            query = self.query_parser(lucene.Version.LUCENE_CURRENT, "text", self.analyzer).parse(query)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

        if self.useExplainQuery:
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,self.max_results)
        else:
            collector=TopScoreDocCollector.create(MAX_RESULTS_RECALL, True)
            self.searcher.search(query, collector)
            hits = collector.topDocs().scoreDocs

##        print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        res=[]


##        if len(hits.scoreDocs) ==0:
##            print "Original query:",original_query
##            print "Query:", query

        for hit in hits:
            doc = self.searcher.doc(hit.doc)
            metadata= json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))
        return res


class luceneRetrievalBoost(luceneRetrieval):
    """
        Use Lucene for retrieval boosting different AZ fields differently
    """

    def __init__(self, indexDir, method, logger=None, use_default_similarity=False):
        luceneRetrieval.__init__(self, indexDir, method, logger, use_default_similarity)

    def generateLuceneQuery(self, query, parameters, test_guid=None):
        """
            Processes input, expands query to inlcude fields and weights
        """
        original_query=query
        query=self.cleanupQuery(query)
        if query=="":
            return None

        self.last_query=query

        query_text=""
        query_words=query.split()

        # this adds to the query a field specially built for this doc
        # in case a doc in the index doesn't have inlink_context but has
        # special inlink_context_special_GUID fields
        if test_guid:
            for param in [p for p in parameters if p in SPECIAL_FIELDS_FOR_TESTS]:
                parameters[context_extract.getFieldSpecialTestName(param, test_guid)]=parameters[param]

        for index,param in enumerate(parameters):
            for qindex,word in enumerate(query_words):

                query_text+=param+":\""+word+"\"^"+str(parameters[param])+" "
                if qindex < len(query_words)-1:
                    query_text+=" OR "

            if index < len(parameters)-1:
                query_text+=" OR "

        return query_text

    def runQuery(self, query, parameters, test_guid):
        """
            Run the query, return a list of tuples (score,metadata) of top docs
        """
        query_text=self.generateLuceneQuery(query, parameters, test_guid)
        try:
            query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
        except:
            # this is to deal with the "too many boolean clauses" exception
            # !TODO implement in Java a query that can deal with this
            try:
                simplified_query_text=" ".join(query.split()[:(1024/len(parameters))])
                query_text=self.generateLuceneQuery(simplified_query_text, parameters, test_guid)
                query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
            except:
                print "Lucene exception:",sys.exc_info()[:2]
                print "Query:",query_text
                return None

        if self.useExplainQuery:
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,self.max_results)
        else:
            collector=TopScoreDocCollector.create(MAX_RESULTS_RECALL, True)
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
            print query_text
            print hits
            print res
            print

##        del hits_list
        del hits
        del query
        del query_text

        return res


class storedFormula:
    """
        Stores a Lucene explanation and makes it easy to set weights on the
        formula post-hoc and recompute
    """
    def __init__(self, formula=None):
        if formula:
            self.formula=formula
        else:
            self.formula={}
        self.round_to_decimal_places=4

    def __getitem__(self, key): return self.formula[key]

    def __setitem__(self, key, item): self.formula[key] = item

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return '.'.join([i, (d+'0'*n)[:n]])

    def fromExplanation(self,explanation):
        """
            Loads the formula from a Lucene explanation
        """
        original_value=explanation.getValue()

        if not explanation.isMatch():
            self.formula={"coord":0,"matches":[]}
            return

        details=explanation.getDetails()
        self.formula["matches"]=[]

        if "weight(" in details[0].getDescription(): # if coord == 1 it is not reported
            matches=details
            self.formula["coord"]=1
        else:
            matches=details[0].getDetails()
            self.formula["coord"]=details[1].getValue()

        for match in matches:
            desc=match.getDescription()
            field=re.match(r"weight\((.*?)\:",desc,re.IGNORECASE)
            # using dicts
##            newMatch={"field":field.group(1)}
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch["qw"]=1.0
##                newMatch["fw"]=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch["qw"]=elements[0].getValue()
##                newMatch["fw"]=elements[1].getValue()
            # using namedtuple
##            newMatch=namedtuple("retrieval_result",["field","qw","fw"])
##            newMatch.field=str(field.group(1))
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch.qw=1.0
##                newMatch.fw=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch.qw=elements[0].getValue()
##                newMatch.fw=elements[1].getValue()

            # using tuple
            field_name=str(field.group(1))
            elem=match.getDetails()[0]
            if "fieldWeight" in elem.getDescription():
                # if the queryWeight is 1, .explain() will not report it
                newMatch=(field_name,1.0,elem.getValue())
            else:
                elements=elem.getDetails()
                newMatch=(field_name,elements[0].getValue(),elements[1].getValue())
            self.formula["matches"].append(newMatch)

        # just checking
##        original_value=self.truncate(original_value,self.round_to_decimal_places)
##        computed_value=self.truncate(self.computeScore(defaultdict(lambda:1),self.round_to_decimal_places))
##        assert(computed_value == original_value)

    def computeScore(self,parameters):
        """
            Simple recomputation of a Lucene explain formula using the values in
            [parameters] as per-field query weights
        """
        match_sum=0.0
        for match in self.formula["matches"]:
##            match_sum+=(match["qw"]*parameters[match["field"]])*match["fw"]
            match_sum+=(match[1]*parameters[match[0]])*match[2]
        total=match_sum*self.formula["coord"]
##        total=self.truncate(total, self.round_to_decimal_places) # x digits of precision
        return total

class precomputedExplainRetrieval(luceneRetrievalBoost):
    """
        Class that runs the explain pipeline and extracts the data from a lucene
        explanation to then be able to run the same retrieval with different
        field boosts.

        All it does is this one-time retrieval + explanation. The further testing
        of parameters is done in measurePrecomputedResolution()
    """

    def __init__(self, indexDir, method, logger=None, use_default_similarity=False):
        luceneRetrievalBoost.__init__(self, indexDir, method, logger, use_default_similarity)

    def precomputeExplain(self, query, parameters, test_guid):
        """
            Runs Lucene retrieval, extracts and stores the explanations in a dict:
                one entry per potential paper, fields to set weights to as sub-entries
        """

        query_text=self.generateLuceneQuery(query, parameters, test_guid)
        try:
            query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

        results=[]

        index=0

        # we first run a search, so that we only need to run the explain pipeline
        # for doc that actually match the query. This is just to speed up
        # whole-collection precomputing of retrieval formulas
        collector=TopScoreDocCollector.create(MAX_RESULTS_RECALL, True)
        self.searcher.search(query, collector)
        hits = collector.topDocs().scoreDocs
        doc_list=[hit.doc for hit in hits]

        for index in doc_list:
            explanation=self.searcher.explain(query,index)

            formula=storedFormula()
            formula.fromExplanation(explanation)

            if formula.formula["coord"] != 0:
                doc = self.searcher.doc(index)
                results.append({"index":index,"guid":doc.get("guid"),"formula":formula.formula})
        return results


def main():
    pass

if __name__ == '__main__':
    main()
