# <description>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import time
from celery.result import ResultSet

import minerva.db.corpora as cp
from minerva.evaluation.keyword_functions import annotateKeywords
from minerva.db.result_store import ElasticResultStorer
from minerva.retrieval.base_retrieval import BaseRetrieval
from minerva.squad.tasks import annotateKeywordsTask
from precomputed_pipeline import PrecomputedPipeline
from minerva.proc.document_features import DocumentFeaturesAnnotator
from minerva.proc.results_logging import ProgressIndicator

from keyword_functions import MISSING_FILES

class KeywordTrainingPipeline(PrecomputedPipeline):
    """
        Modifies PrecomputedPipeline to change what is stored for each precomputed
        result, as all we need to store is:
            query,
            citation,
            context (doc or selection of sentences),
            kws to extract with weights
    """
    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        super(KeywordTrainingPipeline, self).__init__(retrieval_class=retrieval_class, use_celery=use_celery)
        self.current_guid=""
        self.current_doc=None
        self.save_terms=True

    def createWriterInstances(self):
        """
            Initializes the writer instances.
        """
        self.writers["ALL"]=ElasticResultStorer(self.exp["name"],"kw_data", endpoint=cp.Corpus.endpoint)
        if self.options.get("clear_existing_prr_results",False):
            self.writers["ALL"].clearResults()

    def annotateDocuments(self):
        """
            If "run_feature_annotation" in .options is True, it runs the DocumentFeaturesAnnotator
            on each document from which a query was extracted
        """
        if self.options.get("run_feature_annotation",False):

            annotator=DocumentFeaturesAnnotator(self.exp["features_index_name"])
            all_guids=set()
            for query in self.precomputed_queries:
                all_guids.add(query["file_guid"])

            progress=ProgressIndicator(True,len(all_guids),True)

            for guid in all_guids:
                doc=cp.Corpus.loadSciDoc(guid)
                annotator.annotate(doc)
                cp.Corpus.saveSciDoc(doc)
                progress.showProgressReport("Annotating documents")

    def addResult(self, file_guid, precomputed_query, doc_method, retrieved_results):
        """
            Overrides BaseTestingPipeline.addResult so that for each retrieval result
            we actually run .explain() on each item and we store the precomputed
            formula.
        """
        doc_list=[hit[1]["guid"] for hit in retrieved_results]

        # TODO restrict by query types?
##        for zone_type in ["csc_type", "az"]:
##            if precomputed_query.get(zone_type,"") != "":
##                if self.writers[zone_type+"_"+precomputed_query[zone_type].strip()].getResultCount() < self.max_per_class_results:
##                    must_process=True
##                else:
##                    must_process=False
##                    # TODO this is redundant now. Merge this into base_pipeline.py?
##                    print(u"Too many queries of type {} already".format(precomputed_query[zone_type]))
##        if not must_process:
##            return

        if self.current_guid != file_guid:
            self.current_guid=file_guid
            self.current_doc=cp.Corpus.loadSciDoc(file_guid)

        if precomputed_query["match_guid"] not in doc_list:
            doc_list.append(precomputed_query["match_guid"])

        if self.use_celery:
            print("Adding subtask to queue...")
            self.tasks.append(annotateKeywordsTask.apply_async(args=[
                                                 precomputed_query,
                                                 doc_method,
                                                 doc_list,
                                                 self.tfidfmodels[doc_method].index_name,
                                                 self.exp["name"],
                                                 self.exp["experiment_id"],
                                                 self.exp["context_extraction"],
                                                 self.exp["context_extraction_parameter"],
                                                 self.exp["keyword_selection_method"],
                                                 self.exp["max_results_recall"]],
                                                 queue="annotate_keywords"))
        else:
            annotateKeywords(precomputed_query,
                             self.current_doc,
                             doc_method,
                             doc_list,
                             self.tfidfmodels[doc_method],
                             self.writers,
                             self.exp["experiment_id"],
                             self.exp["context_extraction"],
                             self.exp["context_extraction_parameter"],
                             self.exp["keyword_selection_method"],
                             )


    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.

            Should the results be saved?
        """

        with open(r"C:\NLP\PhD\aac\output\missing_files.csv", "w", ) as f:
            f.write("file_guid,match_guid,query,match_title,match_year,in_papers\n")
            for mfile in MISSING_FILES:
                for index,item in enumerate(mfile):
                    try:
                        f.write("\"" + unicode(item).replace("\"","") + "\"")
                    except:
                        f.write(u"<unicode error>")

                    if index < len(mfile)-1:
                        f.write(",")
                    else:
                        f.write("\n")

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

##        for writer in self.writers:
##            self.writers[writer].saveAsJSON(os.path.join(self.exp["exp_dir"],self.writers[writer].table_name+".json"))




def main():
    pass

if __name__ == '__main__':
    main()
