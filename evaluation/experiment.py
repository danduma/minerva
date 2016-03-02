# Experiment class
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import gc,sys,json,os,datetime
from os.path import exists

import minerva.db.corpora as cp

from prebuild import prebuildBOWsForTests

from minerva.evaluation.query_generation import QueryGenerator
from minerva.evaluation.base_pipeline import BaseTestingPipeline
from minerva.evaluation.precomputed_pipeline import PrecomputedPipeline
from minerva.proc.query_extraction import EXTRACTOR_LIST

from results_analysis import saveGraphForResults, makeAllGraphsForExperiment

from minerva.proc.query_extraction import EXTRACTOR_LIST
import minerva.proc.doc_representation as doc_representation
import minerva.evaluation.athar_corpus as athar_corpus
from minerva.proc.general_utils import ensureDirExists
from weight_training import dynamicWeightValues, measureScores

class Experiment:
    """
        Encapsulates an experiment's parameters.

    """
    def __init__(self, experiment, options={}):
        """
            Pass either a dict
        """
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
            if not self.experimentExists(exp_name):
                raise IOError("Experiment file does not exist for "+exp_name)
                return

            self.exp=self.loadExperiment(exp_name)
        else:
            from copy import deepcopy
            self.exp=deepcopy(experiment)
            self.exp["exp_dir"]=os.path.join(cp.Corpus.paths.experiments,self.exp["name"])+os.sep
        ensureDirExists(self.exp["exp_dir"])
        self.options=options

        if cp.Corpus.__class__.__name__ == "ElasticCorpus":
            from elastic_index import ElasticIndexer
            from elastic_retrieval import ElasticRetrievalBoost
            self.indexer=ElasticIndexer()
            self.retrieval_class=ElasticRetrievalBoost
        elif cp.Corpus.__class__.__name__ == "LocalCorpus":
            from lucene_index import LuceneIndexer
            from lucene_retrieval import LuceneRetrievalBoost
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

    def trainWeights(self):
        """
        """
        gc.collect()
        options=self.options

##        if options["run_precompute_retrieval"] or not exists(self.exp["exp_dir"]+"prr_"+self.exp["queries_classification"]+"_"+self.exp["train_weights_for"][0]+".json"):
##            self.query_generator.precomputeQueries(self.exp)
        # Is this where I should be loading the pre-computed formulas?
        # They're already being loaded insinde dynamicWeightValues()

        best_weights={}
        if options.get("override_folds",None):
            self.exp["cross_validation_folds"]=options["override_folds"]

        if options.get("override_metric",None):
            self.exp["metric"]=options["override_metric"]

        numfolds=self.exp.get("cross_validation_folds",2)

        for split_fold in range(numfolds):
            print("\nFold #"+str(split_fold))
            best_weights[split_fold]=dynamicWeightValues(self.exp,split_fold)

        print("Now applying and testing weights...\n")
        measureScores(self.exp, best_weights)

    def run(self):
        """
            Loads a JSON describing an experiment and runs it all
        """
        self.exp["experiment_id"]=datetime.datetime.now().isoformat("/")

        # BIND EXTRACTORS
        for option in self.exp["prebuild_bows"]:
            self.exp["prebuild_bows"][option]["function"]=self.bindFunction(self.exp["prebuild_bows"][option]["function"])

        for option in self.exp["qmethods"]:
            self.exp["qmethods"][option]["extractor"]=self.bindExtractor(self.exp["qmethods"][option]["method"])

        # TEST FILES
        if len(self.exp["test_files"]) > 0:
            cp.Corpus.TEST_FILES=self.exp["test_files"]
        else:
            if self.exp["test_files_condition"] == "":
                raise ValueError("No test_files specified or test_files_condition")
                return
            cp.Corpus.TEST_FILES=cp.Corpus.listPapers(self.exp["test_files_condition"])
            self.exp["test_files"]=cp.Corpus.TEST_FILES

        # PREBUILD BOWS
        if self.exp["full_corpus"]:
            prebuild_list=cp.Corpus.listPapers()
        else:
            prebuild_list=cp.Corpus.listIncollectionReferencesOfList(cp.Corpus.TEST_FILES)
            prebuild_list.extend(cp.Corpus.TEST_FILES)

        if self.options["run_prebuild_bows"] and len(self.exp["prebuild_bows"]) > 0:
            prebuildBOWsForTests(
                self.exp["prebuild_bows"],
                FILE_LIST=prebuild_list,
                force_prebuild=self.options["force_prebuild"],
                rhetorical_annotations=self.exp.get("rhetorical_annotations",[]))

        # BUILD INDEX
        if not self.exp["full_corpus"]:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_indexes"]) > 0:
                self.indexer.buildIndexes(cp.Corpus.TEST_FILES, self.exp["prebuild_indexes"])
        else:
            if self.options["rebuild_indexes"] and len(self.exp["prebuild_general_indexes"]) > 0:
                self.indexer.buildGeneralIndex(cp.Corpus.TEST_FILES,self.exp["prebuild_general_indexes"])

        gc.collect()
        # COMPUTE QUERIES
        if self.exp.get("queries_classification", None) != None:
            queries_file=os.path.join(self.exp["exp_dir"],"queries_by_"+self.exp["queries_classification"]+".json")
        else:
            queries_file=os.path.join(self.exp["exp_dir"],self.exp["precomputed_queries_filename"])

        if self.options["recompute_queries"] or not exists(queries_file):
            self.query_generator.precomputeQueries(self.exp)

        self.exp["precomputed_queries_file_path"]=queries_file
        gc.collect()

        # TESTING PIPELINE
        if self.exp["type"] == "compute_once":
            pipeline=BaseTestingPipeline(retrieval_class=self.retrieval_class)
            pipeline.runPipeline(self.exp)
        elif self.exp["type"] == "train_weights":
            pipeline=PrecomputedPipeline(retrieval_class=self.retrieval_class)
            pipeline.runPipeline(self.exp)
            self.trainWeights()


def main():
    pass

if __name__ == '__main__':
    main()
