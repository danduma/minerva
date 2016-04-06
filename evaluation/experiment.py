# Experiment class
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import gc,json,os,datetime
from os.path import exists

import minerva.db.corpora as cp

from base_prebuild import BasePrebuilder

from minerva.evaluation.query_generation import QueryGenerator
from minerva.evaluation.base_pipeline import BaseTestingPipeline
from minerva.evaluation.precomputed_pipeline import PrecomputedPipeline
from minerva.proc.query_extraction import EXTRACTOR_LIST

##from results_analysis import saveGraphForResults, makeAllGraphsForExperiment
import minerva.proc.doc_representation as doc_representation
import minerva.evaluation.athar_corpus as athar_corpus
from minerva.proc.general_utils import ensureDirExists
from weight_training import WeightTrainer

class Experiment(object):
    """
        Encapsulates an experiment's parameters and global running pipeline.

    """
    def __init__(self, experiment, options={}, use_celery=False):
        """
            :param experiment: either a dict or a file name to load
        """
        self.use_celery=use_celery
        self.load(experiment, options)
        self.query_generator=QueryGenerator()

    def experimentExists(self, filename):
        """
            True if pickled file is found
        """
        return os.path.isfile(cp.Corpus.paths.experiments+filename+".json")

    def loadExperiment(self,exp_name):
        """
            Return a dict with the details for running an experiment
        """
        exp_path=os.path.join(cp.Corpus.paths.experiments,exp_name)
        filename=exp_path+".json"
        assert(os.path.isfile(filename))
        exp=json.load(open(filename,"r"))
        exp["exp_dir"]=os.path.join(exp_path,os.sep)

        return exp

    def load(self, experiment, options):
        """
            if `experiment` is a string, loads the experiment file from the
            experiments directory as specified by the `corpus`.
            if `experiment` is a dict, copies the experiment data from it
        """
        if not isinstance(experiment, dict):
            if not self.experimentExists(experiment):
                raise IOError("Experiment file does not exist for "+experiment)
                return

            self.exp=self.loadExperiment(experiment)
        else:
            from copy import deepcopy
            self.exp=deepcopy(experiment)
            self.exp["exp_dir"]=os.path.join(cp.Corpus.paths.experiments,self.exp["name"])+os.sep
        ensureDirExists(self.exp["exp_dir"])
        self.options=options

        if cp.Corpus.__class__.__name__ == "ElasticCorpus":
            from minerva.retrieval.elastic_index import ElasticIndexer
            from minerva.retrieval.elastic_retrieval import ElasticRetrievalBoost
            self.indexer=ElasticIndexer(use_celery=self.use_celery)
            self.retrieval_class=ElasticRetrievalBoost
        elif cp.Corpus.__class__.__name__ == "LocalCorpus":
            from minerva.retrieval.lucene_index import LuceneIndexer
            from minerva.retrieval.lucene_retrieval import LuceneRetrievalBoost
            self.indexer=LuceneIndexer()
            self.retrieval_class=LuceneRetrievalBoost

    def bindFunction(self,function):
        """
            Binds a name of an extractor function to the actual function
        """
        func=getattr(doc_representation, function, None)
        if not func:
            func=getattr(athar_corpus, function, None)
        assert func
        return func

    def bindExtractor(self,extractor_name):
        """
            Binds a name of an extractor to the actual extractor
        """
        extractor=EXTRACTOR_LIST.get(extractor_name,None)
        if not extractor:
            raise ValueError("Unknown extractor: %s" % extractor_name)
        return extractor

    def selectTestFiles(self):
        """
        """
        if len(self.exp["test_files"]) > 0:
            cp.Corpus.TEST_FILES=self.exp["test_files"]
        else:
            if self.exp["test_files_condition"] == "":
                raise ValueError("No test_files specified or test_files_condition")

            cp.Corpus.TEST_FILES=cp.Corpus.listPapers(self.exp["test_files_condition"])
            if self.exp.get("max_test_files",None):
                cp.Corpus.TEST_FILES=cp.Corpus.TEST_FILES[:self.exp["max_test_files"]]
            self.exp["test_files"]=cp.Corpus.TEST_FILES


    def bindAllExtractors(self):
        """
        """
        for option in self.exp["prebuild_bows"]:
            function_name=self.exp["prebuild_bows"][option]["function"]
            self.exp["prebuild_bows"][option]["function_name"]=function_name
            self.exp["prebuild_bows"][option]["function"]=self.bindFunction(function_name)

        for option in self.exp["qmethods"]:
            self.exp["qmethods"][option]["extractor"]=self.bindExtractor(self.exp["qmethods"][option]["method"])

    def prebuildBOWs(self):
        """
        """
        if self.options["run_prebuild_bows"] and len(self.exp["prebuild_bows"]) > 0:
            if self.exp["full_corpus"]:
                if self.exp.get("index_max_year",None):
                    prebuild_list=cp.Corpus.listPapers("metadata.year:<=%d" % self.exp["index_max_year"])
                else:
                    prebuild_list=cp.Corpus.listPapers()
            else:
                prebuild_list=cp.Corpus.listIncollectionReferencesOfList(cp.Corpus.TEST_FILES)
                prebuild_list.extend(cp.Corpus.TEST_FILES)

            cp.Corpus.ALL_FILES=prebuild_list
            prebuilder=BasePrebuilder(self.use_celery)
            prebuilder.prebuildBOWsForTests(self.exp, self.options)

    def buildIndex(self):
        """
        """
        if not self.exp["full_corpus"]:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_indexes"]) > 0:
                self.indexer.buildIndexes(cp.Corpus.TEST_FILES, self.exp["prebuild_indexes"], self.options)
        else:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_general_indexes"]) > 0:
                self.indexer.buildGeneralIndex(self.exp)

    def computeQueries(self):
        """
        """
        gc.collect()
        queries_file=os.path.join(self.exp["exp_dir"],self.exp["precomputed_queries_filename"])

        if self.options["recompute_queries"] or not exists(queries_file):
            self.query_generator.precomputeQueries(self.exp)

        self.exp["precomputed_queries_file_path"]=queries_file
        gc.collect()


    def runTestingPipeline(self):
        """
        """
        if self.exp["type"] == "compute_once":
            pipeline=BaseTestingPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
            pipeline.runPipeline(self.exp)
        elif self.exp["type"] == "train_weights":
            if self.options.get("run_precompute_retrieval", False):
                pipeline=PrecomputedPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
                pipeline.runPipeline(self.exp)
            weight_trainer=WeightTrainer(self.exp, self.options)
            weight_trainer.trainWeights()
        elif self.exp["type"] in ["", "do_nothing"]:
            return
        else:
            raise NotImplementedError("Unkown experiment type")


    def run(self):
        """
            Loads an experiment and runs it all
        """
        self.exp["experiment_id"]=datetime.datetime.now().isoformat("/")

        # BIND EXTRACTORS
        self.bindAllExtractors()

        # TEST FILES
        self.selectTestFiles()

        # PREBUILD BOWS
        self.prebuildBOWs()

        # BUILD INDEX
        self.buildIndex()

        # COMPUTE QUERIES
        self.computeQueries()

        # TESTING PIPELINE
        self.runTestingPipeline()

def main():
    pass

if __name__ == '__main__':
    main()
