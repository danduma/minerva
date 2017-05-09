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
from minerva.evaluation.keyword_pipeline import KeywordTrainingPipeline

from minerva.proc.query_extraction import EXTRACTOR_LIST

##from results_analysis import saveGraphForResults, makeAllGraphsForExperiment
import minerva.proc.doc_representation as doc_representation
import minerva.evaluation.athar_corpus as athar_corpus
from minerva.proc.general_utils import ensureDirExists
from weight_training import WeightTrainer
from keyword_training import KeywordTrainer

def loadExpFromPath(path):
    """
        Load experinment and options from json
    """
    if path[-1] != os.sep:
        path+=os.sep

    exp=json.load(file(path+"exp.json","r"))
    options=json.load(file(path+"options.json","r"))
    return exp, options

def saveExpToPath(exp, options, path):
    """
        Save experinment and options to json files
    """
    if path[-1] != os.sep:
        path+=os.sep

    exp=json.dump(exp, file(path+"exp.json","w"),indent=3)
    options=json.dump(options, file(path+"options.json","w"),indent=3)


class Experiment(object):
    """
        Encapsulates an experiment's parameters and global running pipeline.

    """
    def __init__(self, experiment, options={}, use_celery=False, process_command_line=True):
        """
            :param experiment: either a dict or a file name to load
            :param use_celery: if true, tasks will not be executed automatically
            but added to the celery queue. See squad.celery_py:app
        """
        self.use_celery=use_celery
        self.load(experiment, options)
        if process_command_line:
            self.processCommandLineArguments()
        self.exp["experiment_id"]=datetime.datetime.now().isoformat("/")
        self.query_generator=QueryGenerator()

    def processCommandLineArguments(self):
        """
            Processes optional command line arguments
        """
        import argparse
        parser = argparse.ArgumentParser(description='Run experiment %s ' % self.exp["name"])
        parser.add_argument("-d", "--exp_dir",  dest='exp_dir', default=None,
                           help='Experiment directory, i.e. where to store cache files and output')
        parser.add_argument("-w", "--train_weights_for", type=str, nargs='+', dest='train_weights_for', default=None,
                           help='List of query classes that we should train weights for, if the experiment is of type weight_training')
        parser.add_argument("-s", "--running_stage", type=int, dest='running_stage', default=None,
                           help='Running stage')

        args = parser.parse_args()
        self.arguments=args.__dict__

        for arg in self.arguments:
            for which in [self.exp, self.options]:
                if arg in which:
                    arg_val=self.arguments[arg]
                    if arg_val:
                        which[arg]=arg_val
                        if arg == "train_weights_for":
                            self.options["clear_existing_prr_results"]=False
                            self.options["run_precompute_retrieval"]=False

        if self.arguments["running_stage"]:
            if self.arguments["running_stage"]==1:              # stage 1: Build BOWs only
                self.options["run_prebuild_bows"]=True
                self.options["overwrite_existing_bows"]=True
                self.options["rebuild_indexes"]=False
                self.options["compute_queries"]=False
                self.options["overwrite_existing_queries"]=False
                self.options["run_precompute_retrieval"]=False
                self.options["run_experiment"]=False
            elif self.arguments["running_stage"]==2:            # stage 2: build the indeces only
                self.options["run_prebuild_bows"]=False
                self.options["overwrite_existing_bows"]=False
                self.options["rebuild_indexes"]=True
                self.options["compute_queries"]=False
                self.options["overwrite_existing_queries"]=False
                self.options["run_precompute_retrieval"]=False
                self.options["run_experiment"]=False
            elif self.arguments["running_stage"]==3:            # stage 3: build the queries only
                self.options["run_prebuild_bows"]=False
                self.options["overwrite_existing_bows"]=False
                self.options["rebuild_indexes"]=False
                self.options["compute_queries"]=True
                self.options["overwrite_existing_queries"]=True
                self.options["run_precompute_retrieval"]=False
                self.options["run_experiment"]=False
            elif self.arguments["running_stage"]==4:            # stage 4: precompute retrieval only
                self.options["run_prebuild_bows"]=False
                self.options["overwrite_existing_bows"]=False
                self.options["rebuild_indexes"]=False
                self.options["compute_queries"]=False
                self.options["overwrite_existing_queries"]=False
                self.options["clear_existing_prr_results"]=True
                self.options["run_precompute_retrieval"]=True
                self.options["run_experiment"]=False
            elif self.arguments["running_stage"]==5:            # stage 5: run pipeline only
                self.options["run_prebuild_bows"]=False
                self.options["overwrite_existing_bows"]=False
                self.options["rebuild_indexes"]=False
                self.options["compute_queries"]=False
                self.options["overwrite_existing_queries"]=False
                self.options["run_precompute_retrieval"]=False
                self.options["run_experiment"]=True
            elif self.arguments["running_stage"]==9:              # stage 9: annotate documents and run precompute retrieval
                self.options["run_feature_annotation"]=True           # (this stage is in fact separate, can be run before the others)
                self.options["run_prebuild_bows"]=False
                self.options["overwrite_existing_bows"]=False
                self.options["rebuild_indexes"]=False
                self.options["compute_queries"]=False
                self.options["overwrite_existing_queries"]=False
                self.options["run_precompute_retrieval"]=True
                self.options["run_experiment"]=False

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
        self.options=options

        if cp.Corpus.__class__.__name__ == "ElasticCorpus":
            from minerva.retrieval.elastic_index import ElasticIndexer
            from minerva.retrieval.elastic_retrieval import ElasticRetrievalBoost
            self.indexer=ElasticIndexer(endpoint=cp.Corpus.endpoint,use_celery=self.use_celery)
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

        with file(os.path.join(self.exp["exp_dir"],"test_guids.txt"),"w") as f:
            for fname in self.exp["test_files"]:
                f.write(fname)
                f.write("\n")

    def setupExperimentDir(self):
        """
            Ensures dir exists, etc.
        """
        self.exp["exp_dir"]=os.path.normpath(os.path.join(cp.Corpus.paths.experiments,self.exp["name"])) + os.sep
        ensureDirExists(self.exp["exp_dir"])

    def bindAllExtractors(self):
        """
            Finds and links the functions (from doc_representation.py, to the "function" key) or extractor instance
            (from query_extraction.EXTRACTOR_LIST to the "extractor" key)
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
        if self.exp["full_corpus"]:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_general_indexes"]) > 0:
                self.indexer.buildGeneralIndex(self.exp, self.options)
        else:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_indeces"]) > 0:
                self.indexer.buildIndexes(cp.Corpus.TEST_FILES, self.exp["prebuild_indeces"], self.options)

    def computeQueries(self):
        """
            Compute or load the queries
        """
        gc.collect()
        queries_file=os.path.join(self.exp["exp_dir"],self.exp["precomputed_queries_filename"])

        if self.options.get("compute_queries",True) and (self.options.get("overwrite_existing_queries",False) or not exists(queries_file)):
            self.query_generator.precomputeQueries(self.exp)

        self.exp["precomputed_queries_file_path"]=queries_file
        gc.collect()


    def runTestingPipeline(self):
        """
            Last step of self.run()
        """
        if self.exp["type"] == "compute_once":
            if self.options.get("run_experiment", True):
                pipeline=BaseTestingPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
                pipeline.runPipeline(self.exp, self.options)
        elif self.exp["type"] == "train_weights":
            if self.options.get("run_precompute_retrieval", False):
                pipeline=PrecomputedPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
                pipeline.runPipeline(self.exp, self.options)
            if self.options.get("run_experiment", True):
                weight_trainer=WeightTrainer(self.exp, self.options)
                weight_trainer.trainWeights()
        elif self.exp["type"] == "extract_kw":
            pipeline=KeywordTrainingPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
            if self.options.get("run_precompute_retrieval", False):
                pipeline.runPipeline(self.exp, self.options)
            if self.options.get("refresh_results_cache", False):
                pipeline.cacheResultsLocally()

            if self.options.get("run_experiment", True):
                kw_trainer=KeywordTrainer(self.exp, self.options)
                kw_trainer.trainExtractors()
        elif self.exp["type"] == "test_kw_selection":
            assert False, "Not implemented yet"
            pipeline=KeywordTrainingPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
            if self.options.get("run_precompute_retrieval", False):
                pipeline.runPipeline(self.exp, self.options)
            if self.options.get("refresh_results_cache", False):
                pipeline.cacheResultsLocally()


        elif self.exp["type"] in ["", "do_nothing"]:
            return
        else:
            raise NotImplementedError("Unkown experiment type")

    def run(self):
        """
            Loads an experiment and runs it all
        """
        self.setupExperimentDir()

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
