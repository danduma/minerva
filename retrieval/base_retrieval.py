# Base retrieval class
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
import re

from proc.query_extraction import getFieldSpecialTestName
from proc.structured_query import StructuredQuery

MAX_RESULTS_RECALL = 200
SPECIAL_FIELDS_FOR_TESTS = ["inlink_context"]


class BaseRetrieval(object):
    """
        Base class for all retrieval wrappers
    """

    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
        pass

        raise NotImplementedError

    def rewriteQuery(self, structured_query, parameters, test_guid=None, include_field=True):
        """
            Modify a ready build LuceneQuery to add other field weights

        """
        if "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        original_query = structured_query
        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        lucene_query = ""

        # this adds to the query a field specially built for this doc
        # in case a doc in the index doesn't have inlink_context but has
        # special inlink_context_special_GUID fields
        if test_guid:
            for param in [p for p in parameters if p in SPECIAL_FIELDS_FOR_TESTS]:
                parameters[getFieldSpecialTestName(param, test_guid)] = parameters[param]

        for index, param in enumerate(parameters):
            for qindex, token in enumerate(structured_query):

                t_boost = token.boost
                t_count = token.count

                if t_boost is None:
                    print("NULL! ")
                    print(token, parameters[param], token.boost, token.count)
                    t_boost = 0
                if t_count is None:
                    print("NULL! ")
                    print(token, parameters[param], token.boost, token.count)
                    t_count = 0

                boost = parameters[param] * t_boost * t_count

                # TODO proper computing of the boost formula. Different methods?

                bool = token.bool or ""

                if boost == 0.0:
                    continue
                if include_field:
                    lucene_query += bool + param + ":\"" + token.token + "\"^" + str(boost) + " "
                else:
                    lucene_query += bool + " \"" + token.token + "\"^" + str(boost) + " "

                if qindex < len(structured_query) - 1:
                    lucene_query += " OR "

            if index < len(parameters) - 1:
                lucene_query += " OR "

        lucene_query = re.sub(r"\s+", " ", lucene_query)
        lucene_query = lucene_query.replace(" OR OR", " OR")
        lucene_query = re.sub(r"\s+", " ", lucene_query)
        if lucene_query.endswith("OR"):
            lucene_query = lucene_query[:-3]
        return lucene_query

    def runQuery(self, structured_query, max_results=MAX_RESULTS_RECALL):
        """
        """
        raise NotImplementedError

    def formulaFromExplanation(self, query, doc_id):
        """
            Runs .explain() for one query/doc pair, generates and returns a \
            StoredFormula instance from it

            :param query: depends on the descendant classes. Whatever can be run
                directly. For direct Lucene, that will be a string in the
                Lucene query language, for Elastic, that will be a DSL query
            :param doc_id: id of document to run .explain() for
            :returns:
        """
        raise NotImplementedError


def main():
    pass


if __name__ == '__main__':
    main()
