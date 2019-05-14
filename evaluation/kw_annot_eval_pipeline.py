# Pipeline that generates annotated contexts with the selected keywords to extract
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import os

import db.corpora as cp
from evaluation.keyword_functions import runVariousKeywordSelections, ALL_CONTEXTS, SPECIAL_EXPORT
from db.result_store import ElasticResultStorer
from retrieval.base_retrieval import BaseRetrieval
from evaluation.keyword_annotation_pipeline import KeywordAnnotationPipeline
from ml.keyword_support import saveOfflineKWSelectionTraceToCSV
from evaluation.weight_functions import addExtraWeights
from evaluation import kw_selector_classes
from copy import deepcopy


class KeywordSelectionEvaluationPipeline(KeywordAnnotationPipeline):
    """
        Pipeline that tests different keyword selection algorithms

        Modifies PrecomputedPipeline to change what is stored for each precomputed
        result, as all we need to store is:
            query,
            citation,
            context (doc or selection of sentences),
            kws to extract with weights
    """

    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        super(KeywordSelectionEvaluationPipeline, self).__init__(retrieval_class=retrieval_class, use_celery=use_celery)
        self.current_guid = ""
        self.current_doc = None
        self.save_terms = True
        self.writer_name = "kw_selection_scores"

    def createWriterInstances(self):
        """
            Initializes the writer instances.
        """
        self.writers["ALL"] = ElasticResultStorer(self.exp["name"], self.writer_name, endpoint=cp.Corpus.endpoint)
        if self.options.get("clear_existing_prr_results", False):
            self.writers["ALL"].clearResults()

        self.selectors = deepcopy(self.exp["keyword_selection_parameters"])
        for entry in self.selectors:
            self.selectors[entry]["class_name"] = self.selectors[entry]["class"]
            self.selectors[entry]["class"] = getattr(kw_selector_classes, self.selectors[entry]["class"], None)
            self.selectors[entry]["instance"] = self.selectors[entry]["class"](entry)


    def addResult(self, file_guid, precomputed_query, doc_method, retrieved_results):
        """
            This is where we select the top keywords for a query/citation based on
            the retrieved results and the score of keywords for the document's match

            :param file_guid: guid of the file the citation originates from
            :param precomputed_query: dict with info about the query that has already been extracted
        """
        doc_list = [hit[1]["guid"] for hit in retrieved_results]

        # TODO restrict by query types?


        if self.current_guid != file_guid:
            self.current_guid = file_guid
            self.current_doc = cp.Corpus.loadSciDoc(file_guid)

        # # Add missing guid to the end of the doc_list. This puts a cap on the scores, as the 201st paper will always be the right one.
        # TODO: how much did this impact scores? Why doesn't it impact the other measures?
        # if precomputed_query["match_guid"] not in doc_list:
        #     doc_list.append(precomputed_query["match_guid"])

        weights = addExtraWeights(self.main_all_doc_methods[doc_method]["runtime_parameters"], self.exp)

        if self.exp.get("expand_match_guids", False):
            expanded_guids = cp.Corpus.doc_sim.expandRelevantPapers(precomputed_query["match_guids"],
                                                                    self.exp.get("match_guid_expansion_threshold", 0.2),
                                                                    self.exp.get("match_guid_expansion_max_add", 5),
                                                                    )
            precomputed_query["match_guids"].extend(expanded_guids)

        if self.use_celery:
            print("Adding subtask to queue...")
            # self.tasks.append(annotateKeywordsTask.apply_async(args=[
            #     precomputed_query,
            #     doc_method,
            #     doc_list,
            #     self.retrieval_models[doc_method].index_name,
            #     self.exp["name"],
            #     self.exp["experiment_id"],
            #     self.exp["context_extraction"],
            #     self.exp["context_extraction_parameter"],
            #     self.exp["keyword_selection_parameters"],
            #     self.exp["max_results_recall"],
            #     weights,
            # ],
            #     queue="annotate_keywords"))
        else:
            runVariousKeywordSelections(precomputed_query,
                                        self.current_doc,
                                        doc_method,
                                        doc_list,
                                        self.retrieval_models[doc_method],
                                        self.writers,
                                        self.exp["experiment_id"],
                                        self.exp["context_extraction"],
                                        self.exp["context_extraction_parameter"],
                                        self.selectors,
                                        weights,
                                        self.annotator
                                        )

            if SPECIAL_EXPORT:
                from models.keyword_features import saveFeatureData
                saveFeatureData(ALL_CONTEXTS, "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/thesis_example_selection.json")


    def saveResultsAndCleanUp(self):
        super().saveResultsAndCleanUp()
        for entry in self.selectors:
            self.selectors[entry]["instance"].saveResults(os.path.join(
                self.exp["exp_dir"],
                "output_%s.txt" % self.selectors[entry][
                    "class_name"]))

    def exportAnnotationResults(self):
        # time.sleep(15) # wait for elastic to index everything

        saveOfflineKWSelectionTraceToCSV(self.writer_name,
                                         os.path.join(self.exp["exp_dir"], "cache"),
                                         self.exp["exp_dir"])


def main():
    pass


if __name__ == '__main__':
    main()
