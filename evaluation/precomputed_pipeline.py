# Pipeline that precomputes the retrieval results that can be used for weight training afterwards
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import json, os, time

from base_pipeline import BaseTestingPipeline
from minerva.retrieval.base_retrieval import BaseRetrieval

from celery.result import ResultSet

from minerva.db.result_store import createResultStorers
from precompute_functions import addPrecomputeExplainFormulas
from minerva.multi.tasks import precomputeFormulasTask

class PrecomputedPipeline(BaseTestingPipeline):
    """
        Pipeline for training weights. Queries are run once, the explanation of
        each result is stored and weights are trained.
    """

    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        super(PrecomputedPipeline, self).__init__(retrieval_class=retrieval_class, use_celery=use_celery)
        self.writers={}


    def addResult(self, file_guid, precomputed_query, doc_method, retrieved_results):
        """
            Overrides BaseTestingPipeline.addResult so that for each retrieval result
            we actually run .explain() on each item and we store the precomputed
            formula.
        """
        doc_list=[hit[1]["guid"] for hit in retrieved_results]

        for zone_type in ["csc_type", "az"]:
            if precomputed_query.get(zone_type,"") != "":
                if self.writers[zone_type+"_"+precomputed_query[zone_type].strip()].getResultCount() < self.max_per_class_results:
                    must_process=True
                else:
                    must_process=False
                    # TODO this is redundant now. Merge this into base_pipeline.py?
                    print(u"Too many queries of type {} already".format(precomputed_query[zone_type]))
##                  assert(False)

        if not must_process:
            return

        if self.use_celery:
            print("Adding subtask to queue...")
            self.tasks.append(precomputeFormulasTask.apply_async(args=[
                                                 precomputed_query,
                                                 doc_method,
                                                 doc_list,
                                                 self.tfidfmodels[doc_method].index_name,
                                                 self.exp["name"],
                                                 self.exp["experiment_id"],
                                                 self.exp["max_results_recall"]],
                                                 queue="precompute_formulas"))
        else:
            addPrecomputeExplainFormulas(precomputed_query,
                                         doc_method,
                                         doc_list,
                                         self.tfidfmodels[doc_method],
                                         self.writers,
                                         self.exp["experiment_id"],
                                         )


    def createWriterInstances(self):
        """
            Initializes the writer instances.
        """
        self.writers=createResultStorers(self.exp["name"],
                                   self.exp.get("random_zoning", False),
                                   self.options.get("clear_existing_prr_results", False))


    def loadQueriesAndFileList(self):
        """
            Loads the precomputed queries and the list of test files to process.
        """
        precomputed_queries_file_path=self.exp.get("precomputed_queries_file_path",None)
        if not precomputed_queries_file_path:
            precomputed_queries_file_path=os.path.join(self.exp["exp_dir"],self.exp.get("precomputed_queries_filename","precomputed_queries.json"))

        if "ALL" in self.exp.get("queries_to_process",["ALL"]):
            self.precomputed_queries=json.load(open(precomputed_queries_file_path,"r"))#[:1]
##            precomputed_queries=json.load(open(self.exp["exp_dir"]+"precomputed_queries.json","r"))
        else:
            queries_filename="queries_by_"+self.exp["queries_classification"]+".json"
            queries_by_zone=json.load(open(self.exp["exp_dir"]+queries_filename,"r"))
            self.precomputed_queries=[]
            for zone in queries_by_zone[self.exp["queries_to_process"]]:
                self.precomputed_queries.extend(queries_by_zone[zone])

        print("Total precomputed queries: ",len(self.precomputed_queries))

        files_dict_filename=os.path.join(self.exp["exp_dir"],self.exp.get("files_dict_filename","files_dict.json"))
        self.files_dict=json.load(open(files_dict_filename,"r"))
        self.files_dict["ALL_FILES"]={}

        assert self.exp["name"] != "", "Experiment needs a name!"
        self.createWriterInstances()

    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.
        """
        if self.use_celery:
            print("Waiting for tasks to complete...")
            res=ResultSet(self.tasks)
            while not res.ready():
                try:
                    time.sleep(7)
                except KeyboardInterrupt:
                    print("Cancelled waiting")
                    break
            print("All tasks finished.")

        for writer in self.writers:
            self.writers[writer].saveAsJSON(os.path.join(self.exp["exp_dir"],self.writers[writer].table_name+".json"))


def main():
    pass

if __name__ == '__main__':
    main()
