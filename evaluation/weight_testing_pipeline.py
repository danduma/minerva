# Pipeline that tests the pre-trained weights for different zones
#
# Copyright:   (c) Daniel Duma 2018
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import json, os, time

from .base_pipeline import BaseTestingPipeline
from retrieval.base_retrieval import BaseRetrieval

from celery.result import ResultSet

import csv


class WeightTestingPipeline(BaseTestingPipeline):
    """
        Pipeline for testing weights. The weights are loaded from CSV files and applied to each query.
    """

    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        super(WeightTestingPipeline, self).__init__(retrieval_class=retrieval_class, use_celery=use_celery)
        self.writers = {}

    def createWriterInstances(self):
        """
            Initializes the writer instances.
        """
        # self.writers = createResultStorers(self.exp["name"],
        #                                    self.exp.get("random_zoning", False),
        #                                    self.options.get("clear_existing_prr_results", False))

    def loadWeights(self, weights_filenames):
        """

        :return:
        """
        weights = {}

        for filename in weights_filenames:
            reader = csv.reader(open(filename, "r"), delimiter="\t")

            columns = []

            for index, row in enumerate(reader):
                if index == 0:
                    columns = row
                    while columns[0] == "":
                        columns = columns[1:]
                    continue

                qtype = row[0]
                weights[qtype] = weights.get(qtype, {})

                for row_index in range(1, len(row)):
                    weights[qtype][columns[row_index - 1]] = float(row[row_index])
        return weights

    def initializePipeline(self):
        """
            Whatever needs to happen before we start the pipeline: inializing
            connections, VMs, whatever.

        """
        for method in self.exp["doc_methods"]:
            if "preset_runtime_weights_files" in self.exp["doc_methods"][method]:
                self.exp["doc_methods"][method]["preset_runtime_weights"] = self.loadWeights(
                    self.exp["doc_methods"][method]["preset_runtime_weights_files"])

    def setRuntimeParameters(self, method, precomputed_query):
        """
            Sets the runtime parameters to use depending on the method and query

        """
        preset_weights = self.main_all_doc_methods[method].get("preset_runtime_weights", [])
        if precomputed_query["az"] in preset_weights:
            query_type = precomputed_query["az"]
        elif precomputed_query["csc_type"] in preset_weights:
            query_type = precomputed_query["csc_type"]
        else:
            print("Query type %s %s not found in preset_weights" % (precomputed_query["az"], precomputed_query["csc_type"]))
            raise ValueError

        return preset_weights[query_type]

    def loadQueriesAndFileList(self):
        """
            Loads the precomputed queries and the list of test files to process.
        """
        precomputed_queries_file_path = self.exp.get("precomputed_queries_file_path", None)
        if not precomputed_queries_file_path:
            precomputed_queries_file_path = os.path.join(self.exp["exp_dir"],
                                                         self.exp.get("precomputed_queries_filename",
                                                                      "precomputed_queries.json"))

        if "ALL" in self.exp.get("queries_to_process", ["ALL"]):
            self.precomputed_queries = json.load(open(precomputed_queries_file_path, "r"))  # [:1]
            self.precomputed_queries = self.precomputed_queries[self.options.get("run_query_start_at", 0):]
        ##            precomputed_queries=json.load(open(self.exp["exp_dir"]+"precomputed_queries.json","r"))
        else:
            queries_filename = "queries_by_" + self.exp["queries_classification"] + ".json"
            queries_by_zone = json.load(open(self.exp["exp_dir"] + queries_filename, "r"))
            self.precomputed_queries = []
            for zone in queries_by_zone[self.exp["queries_to_process"]]:
                self.precomputed_queries.extend(queries_by_zone[zone])

        print("Total precomputed queries: ", len(self.precomputed_queries))

        files_dict_filename = os.path.join(self.exp["exp_dir"],
                                           self.exp.get("files_dict_filename", "files_dict.json"))
        self.files_dict = json.load(open(files_dict_filename, "r"))
        self.files_dict["ALL_FILES"] = {}

        assert self.exp["name"] != "", "Experiment needs a name!"
        self.createWriterInstances()

    def addResult(self, file_guid, precomputed_query, doc_method, retrieved_results):
        """
            Overrides BaseTestingPipeline.addResult so that for each retrieval result
            we actually run .explain() on each item and we store the precomputed
            formula.
        """
        # doc_list = [hit[1]["guid"] for hit in retrieved_results]

        # for zone_type in ["csc_type", "az"]:
        #     if precomputed_query.get(zone_type, "") != "":
        #         if self.writers[zone_type + "_" + precomputed_query[
        #             zone_type].strip()].getResultCount() < self.max_per_class_results:
        #             must_process = True
        #         else:
        #             must_process = False
        #             # TODO this is redundant now. Merge this into base_pipeline.py?
        #             print(u"Too many queries of type {} already".format(precomputed_query[zone_type]))
        # ##                  assert(False)
        #
        # if not must_process:
        #     return

        result_dict = self.newResultDict(file_guid, precomputed_query, doc_method)
        self.logger.measureScoreAndLog(retrieved_results, precomputed_query["citation_multi"], result_dict)

        # if result_dict["mrr_score"] < 0.1:
        #     print("FUCK. FIX THIS", result_dict["mrr_score"], doc_method)

    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.
        """
        self.logger.showFinalSummary()


def main():
    pass


if __name__ == '__main__':
    main()
