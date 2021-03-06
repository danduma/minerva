# Pipeline that generates annotated contexts with the selected keywords to extract
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import time, os
from celery.result import ResultSet

import db.corpora as cp
from evaluation.keyword_functions import annotateKeywords
from db.result_store import ElasticResultStorer, ResultDiskReader
from proc.general_utils import ensureDirExists
from retrieval.base_retrieval import BaseRetrieval
from multi.tasks import annotateKeywordsTask
from evaluation.precomputed_pipeline import PrecomputedPipeline
from ml.document_features import DocumentFeaturesAnnotator
# from proc.results_logging import ProgressIndicator
from tqdm import tqdm
from evaluation.keyword_functions import MISSING_FILES
from evaluation.weight_functions import addExtraWeights
import six


class KeywordTrainingPipeline(PrecomputedPipeline):
    """
        Pipeline that generates the annotated data for training/testing keyword
        extractors

        Modifies PrecomputedPipeline to change what is stored for each precomputed
        result, as all we need to store is:
            query,
            citation,
            context (doc or selection of sentences),
            kws to extract with weights
    """

    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        super(KeywordTrainingPipeline, self).__init__(retrieval_class=retrieval_class, use_celery=use_celery)
        self.current_guid = ""
        self.current_doc = None
        self.save_terms = True

    def initializePipeline(self):

        super(KeywordTrainingPipeline, self).initializePipeline()

        self.annotator = DocumentFeaturesAnnotator(self.exp["features_index_name"])

    def createWriterInstances(self):
        """
            Initializes the writer instances.
        """
        self.writers["ALL"] = ElasticResultStorer(self.exp["name"], "kw_data", endpoint=cp.Corpus.endpoint)
        if self.options.get("clear_existing_prr_results", False):
            self.writers["ALL"].clearResults()

    def annotateDocuments(self):
        """
            If "run_feature_annotation" in .options is True, it runs the DocumentFeaturesAnnotator
            on each document from which a query was extracted
        """
        if self.options.get("run_feature_annotation", False):
            annotator = DocumentFeaturesAnnotator(self.exp["features_index_name"],
                                                  field_name=self.exp.get("features_field_name", "text"),
                                                  experiment=self.exp)
            all_guids = set()
            for query in self.precomputed_queries:
                all_guids.add(query["file_guid"])

            print("Annotating %d documents" % len(all_guids))
            # progress = ProgressIndicator(True, len(all_guids), True)

            pbar = tqdm(all_guids)
            for guid in pbar:
                pbar.set_description("Processing %s" % guid)
                doc = cp.Corpus.loadSciDoc(guid)
                annotator.annotate_scidoc(doc)
                cp.Corpus.saveSciDoc(doc)
                # print(guid,"done")
                # progress.showProgressReport("Annotating documents")

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

        weights = addExtraWeights(self.main_all_doc_methods[doc_method]["runtime_parameters"], self.exp)

        if self.use_celery:
            print("Adding subtask to queue...")
            self.tasks.append(annotateKeywordsTask.apply_async(args=[
                precomputed_query,
                doc_method,
                doc_list,
                self.retrieval_models[doc_method].index_name,
                self.exp["name"],
                self.exp["experiment_id"],
                self.exp["context_extraction"],
                self.exp["context_extraction_parameter"],
                self.exp["keyword_selector"],
                self.exp["keyword_selection_parameters"],
                self.exp["max_results_recall"],
                weights,
            ],
                queue="annotate_keywords"))
        else:
            annotateKeywords(precomputed_query,
                             self.current_doc,
                             doc_method,
                             doc_list,
                             self.retrieval_models[doc_method],
                             self.writers,
                             self.exp["experiment_id"],
                             self.exp["context_extraction"],
                             self.exp["context_extraction_parameter"],
                             self.exp["keyword_selector"],
                             self.exp["keyword_selection_parameters"],
                             weights,
                             self.annotator
                             )

    def saveMissingFiles(self):
        """
            Saves a list of all missing files in the output directory
        """
        file_dir=os.path.join(self.exp["exp_dir"], "output")
        ensureDirExists(file_dir)
        file_path = os.path.join(file_dir, "missing_files.csv")

        with open(file_path, "w", ) as f:
            f.write("file_guid,match_guid,query,match_title,match_year,in_papers\n")
            for mfile in MISSING_FILES:
                for index, item in enumerate(mfile):
                    try:
                        f.write("\"" + six.text_type(item).replace("\"", "") + "\"")
                    except:
                        f.write(u"<unicode error>")

                    if index < len(mfile) - 1:
                        f.write(",")
                    else:
                        f.write("\n")

    def cacheResultsLocally(self):
        """
            Downloads all results from elastic to json in the experiment directory
        """
        reader = ResultDiskReader(self.writers["ALL"], os.path.join(self.exp["exp_dir"], "cache"))
        reader.emptyCache()
        reader.cacheAllItems()

    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.

            Should the results be saved?
        """
        self.cacheResultsLocally()

        if self.use_celery:
            print("Waiting for tasks to complete...")
            res = ResultSet(self.tasks)
            while not res.ready():
                try:
                    time.sleep(7)
                except KeyboardInterrupt:
                    print("Cancelled waiting")
                    break
            print("All tasks finished.")

        if self.options.get("list_missing_files", False):
            self.saveMissingFiles()


##        for writer in self.writers:
##            self.writers[writer].saveAsJSON(os.path.join(self.exp["exp_dir"],self.writers[writer].table_name+".json"))

def main():
    pass


if __name__ == '__main__':
    main()
