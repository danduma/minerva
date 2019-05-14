from proc.results_logging import measureScores
from proc.structured_query import StructuredQuery
from copy import deepcopy

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
    kw_counts = {x[0]: x[1] for x in precomputed_query["structured_query"]}

    query = deepcopy(precomputed_query)
    query["structured_query"] = StructuredQuery([[x[0], kw_counts.get(x[0], 0), 1] for x in selected_keywords])
    kw_data["kw_selection_scores"] = runAndMeasureOneQuery(query, weights, retrieval_model)

    query = deepcopy(precomputed_query)
    query["structured_query"] = StructuredQuery([[x[0], kw_counts.get(x[0], 0), x[1]] for x in selected_keywords])
    kw_data["kw_selection_weight_scores"] = runAndMeasureOneQuery(query, weights, retrieval_model)