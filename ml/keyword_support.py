import os

import pandas as pd
from db.result_store import OfflineResultReader
from models.keyword_features import saveFeatureData


def saveKeywordSelectionScores(reader, exp_dir):
    """
        Saves a CSV to measure the performance of keyword selection
    """

    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        return {
            "precision_score": kw_data["precision_score"],
            "mrr_score": kw_data["mrr_score"],
            "rank": kw_data["rank"],
            "ndcg_score": kw_data["ndcg_score"],

            "precision_score_kw": kw_data["kw_selection_scores"]["precision_score"],
            "mrr_score_kw": kw_data["kw_selection_scores"]["mrr_score"],
            "rank_kw": kw_data["kw_selection_scores"]["rank"],
            "ndcg_score_kw": kw_data["kw_selection_scores"]["ndcg_score"],

            "precision_score_kw_weight": kw_data["kw_selection_weight_scores"]["precision_score"],
            "mrr_score_kw_weight": kw_data["kw_selection_weight_scores"]["mrr_score"],
            "rank_kw_weight": kw_data["kw_selection_weight_scores"]["rank"],
            "ndcg_score_kw_weight": kw_data["kw_selection_weight_scores"]["ndcg_score"],
        }

    lines = []
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data = pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir, "kw_selection_scores.csv"))


def saveAllKeywordSelectionTrace(reader, exp_dir):
    """
        Saves a CSV to trace everything that happened, inspect the dataset
    """

    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        context = kw_data["context"]
        if isinstance(context, list):
            context = u" ".join([s["text"] for s in context])

        res = {
            "cit_ids": kw_data["cit_ids"],
            "cit_multi": kw_data["cit_multi"],
            "context": context,
            "file_guid": kw_data.get("file_guid", ""),
            "match_guids": kw_data["match_guids"],
            "best_kws": [kw[0] for kw in kw_data["best_kws"]],

            "precision_score": kw_data["precision_score"],
            "mrr_score": kw_data["mrr_score"],
            "rank": kw_data["rank"],
            "ndcg_score": kw_data["ndcg_score"],

            "precision_score_kw": kw_data["kw_selection_scores"]["precision_score"],
            "mrr_score_kw": kw_data["kw_selection_scores"]["mrr_score"],
            "rank_kw": kw_data["kw_selection_scores"]["rank"],
            "ndcg_score_kw": kw_data["kw_selection_scores"]["ndcg_score"],

            "precision_score_kw_weight": kw_data["kw_selection_weight_scores"]["precision_score"],
            "mrr_score_kw_weight": kw_data["kw_selection_weight_scores"]["mrr_score"],
            "rank_kw_weight": kw_data["kw_selection_weight_scores"]["rank"],
            "ndcg_score_kw_weight": kw_data["kw_selection_weight_scores"]["ndcg_score"],

            "best_kws_weight": kw_data["best_kws"],

        }
        if "keyword_selection_entry" in kw_data:
            res["keyword_selection_entry"] = kw_data["keyword_selection_entry"]
        return res

    lines = []
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data = pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir, "kw_selection_trace.csv"), encoding="utf-8")


def saveOfflineKWSelectionTraceToCSV(reader_name, cache_dir, output_dir):
    """
    Assumes all data has been cached locally, processes and generates CSV from it

    :param cache_dir: where the cache was saved
    :param output_dir: directory where the CSV output will be saved
    :return:
    """

    reader = OfflineResultReader(reader_name, cache_dir)
    ##            print("\n From OfflineResultReader:",listAllKeywordsToExtractFromReader(self.reader))
    ##            saveKeywordSelectionScores(self.reader, self.exp["exp_dir"])
    saveAllKeywordSelectionTrace(reader, output_dir)


def saveOfflineFeaturesToJson(reader_name, cache_dir, filename):
    reader = OfflineResultReader(reader_name, cache_dir)
    saveFeatureData(reader, filename)