from models.keyword_features import tokenWeight
from proc.results_logging import measureScores
from proc.structured_query import StructuredQuery
from copy import deepcopy
import re


def runAndMeasureOneQuery(query, weights, retrieval_model):
    """
        Runs an actual elastic query. Returns a dict of scores given a precomputed_query,
        weights, the retrieval_model in use and the cit dict

    """
    retrieved_results = retrieval_model.runQuery(query, weights)
    new_scores = {}  # need to pass a dict to measureScores(), results will be stored there
    doc_list = [hit[1]["guid"] for hit in retrieved_results]
    measureScores(doc_list, query["match_guids"], new_scores)
    return new_scores


def getCountsInQuery(precomputed_query, selected_keywords):
    res = {x[0]: x[1] for x in precomputed_query["structured_query"]}
    for kw in selected_keywords:
        if " " in kw[0]:
            pattern = kw[0].replace(" ", r"\W+")
            all_matches = re.findall(pattern, precomputed_query["query_text"])
            count = len(all_matches)
            res[kw[0]] = count
    return res


def runQueryAndMeasureKeywordSelection(precomputed_query, selected_keywords, retrieval_model, weights, kw_data):
    """
        Run queries to compute scores for the provided keywords, adds them to the [kw_data] dict

        :param precomputed_query: original precomputed_query as provided by testing pipeline
        :param selected_keywords: tuples of (keyword,weight) coming out of exp["keyword_selector"].selectKeywords()
        :param retrieval_model: retrieval instance (e.g. ElasticRetrievalBoost)
        :param cit: citation dict
        :param weights: the weights used for retrieval at this point, configurable in the experiment
        :param kw_data: dict that will eventually be stored, containing the precomputed_query, selected kws, etc.
    """
    kw_counts = getCountsInQuery(precomputed_query, selected_keywords)

    # print("Query:", precomputed_query["query_text"])
    query = deepcopy(precomputed_query)

    # StructuredToken(token, count, boost, bool, field, distance)
    query["structured_query"] = StructuredQuery(
        [{"token": kw[0], "count": kw_counts.get(kw[0], 0), "boost": 1} for kw in selected_keywords])
    kw_data["kw_selection_scores"] = runAndMeasureOneQuery(query, weights, retrieval_model)

    # if any([" " in kw[0] for kw in selected_keywords]):
    #     print(precomputed_query["file_guid"],precomputed_query["citation_id"])
    #     print(precomputed_query["vis_text"])
    #     print(precomputed_query["keyphrases"])
    #     print("\n\n")

    # if precomputed_query["file_guid"] == 'b884c939-144c-4d95-9d30-097f8b83e1d3' and precomputed_query["citation_id"]=='cit9':
    #     print("stop")

    query = deepcopy(precomputed_query)
    query["structured_query"] = StructuredQuery(
        [{"token": kw[0], "count": kw_counts.get(kw[0], 0), "boost": tokenWeight(kw)} for kw in selected_keywords])
    kw_data["kw_selection_weight_scores"] = runAndMeasureOneQuery(query, weights, retrieval_model)
