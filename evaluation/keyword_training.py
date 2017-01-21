# Weight Training pipeline
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import  gc, random, os
from copy import deepcopy
from collections import defaultdict
from sklearn import cross_validation
import pandas as pd

import minerva.db.corpora as cp
from minerva.proc.results_logging import ResultsLogger
##from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from base_pipeline import getDictOfTestingMethods
from weight_functions import runPrecomputedQuery
from minerva.db.result_store import ElasticResultStorer, ResultIncrementalReader, ResultDiskReader


GLOBAL_FILE_COUNTER=0

def baseline_score(results):
    """
        Takes some retrieval results and computes the baseline score for them

        TODO store this score from the retrieval?
    """
    pass


def buildTrainingExample(query):
    """
        Given a pre-extracted query (using a QueryExtractor and presumably stored on disk),
        run the query, store the results from .explain()

    """


class KeywordTrainer(object):
    """
        This class encapsulates the training and testing of a keyword extractor,
        using k-fold cross-validation
    """
    def __init__(self, exp, options):
        """
        """
        self.exp=exp
        self.options=options
        self.all_doc_methods={}

    def loadPrecomputedFormulas(self):
        """
            Loads the previously computed retrieval results, including query, etc.
        """
        prr=ElasticResultStorer(self.exp["name"],"prr_"+self.exp["queries_classification"], cp.Corpus.endpoint)
        reader=ResultDiskReader(prr, cache_dir=os.path.join(self.exp["exp_dir"], "cache"), max_results=self.exp.get("max_per_class_results",1000))
        reader.bufsize=30
        return reader


    def trainExtractor(self, split_fold):
        """
            Train an extractor for the given fold
        """
        all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
        annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"] in ["annotated_boost"]]

        numfolds=self.exp.get("cross_validation_folds",2)

        retrieval_results=self.loadPrecomputedFormulas()
        if len(retrieval_results) == 0:
            raise ValueError("No precomputed formulas")

        if len(retrieval_results) < numfolds:
            raise ValueError("Number of results is smaller than number of folds")

        cv = cross_validation.KFold(len(retrieval_results), n_folds=numfolds, shuffle=False, random_state=None) # indices=True, k=None
        cv=[k for k in cv]

        traincv, testcv=cv[split_fold]
        if isinstance(retrieval_results, ResultIncrementalReader):
            train_set=retrieval_results.subset(traincv)
        elif isinstance(retrieval_results, list):
            train_set=[retrieval_results[i] for i in traincv]
        else:
            raise ValueError("Unkown class of results")
##            train_set=retrieval_results.subset(traincv)
##            train_set=[retrieval_results[i] for i in traincv]
        if len(train_set) == 0:
            print("Training set len is 0!")
            return defaultdict(lambda:1)

        print("Training for %d/%d citations " % (len(train_set),len(retrieval_results)))
        trained_models={}
        for method in all_doc_methods:
            res={}
            # what to do with the runtime_parameters?
##            all_doc_methods[method]["runtime_parameters"]=weights
            trained_models[method]=TFIDFKeywordExtractor()
            trained_models[method].train(train_set)

        return trained_models


    def measureScoresOfKeywords(self, trained_extractors):
        """
            Using precomputed keywords from another split set, apply and report score
        """
        numfolds=self.exp.get("cross_validation_folds",2)

        results=[]
        fold_results=[]
        metrics=["avg_mrr","avg_ndcg", "avg_precision","precision_total"]

        print("Experiment:",self.exp["name"])
        print("Metric:",self.exp["metric"])
        print("Weight movements:",self.exp.get("movements",None))

        for split_fold in range(numfolds):
            extractor=trained_extractors[split_fold]
            improvements=[]
            better_zones=[]
            better_zones_details=[]


            # Need to run retrieval and measure scores again, using the keywords it extracts

            if len(retrieval_results) == 0:
                continue

            if len(retrieval_results) < numfolds:
                print("Number of results is smaller than number of folds")
                continue

            cv = cross_validation.KFold(len(retrieval_results), n_folds=numfolds, shuffle=False, random_state=None)
            cv=[k for k in cv] # run the generator
            traincv, testcv=cv[split_fold]
            if isinstance(retrieval_results, ResultIncrementalReader):
                test_set=retrieval_results.subset(testcv)
            elif isinstance(retrieval_results, list):
                test_set=[retrieval_results[i] for i in testcv]
            else:
                raise ValueError("Unkown class of results")

            for method in keywords:
                keywords_baseline=addExtrakeywords({x:1 for x in self.all_doc_methods[method]["runtime_parameters"]}, self.exp)

                query_type=None # TODO extract query_type from stored precomputed query

                scores=self.measurePrecomputedResolution(test_set, method, keywords_baseline, query_type)
                baseline_score=scores[0][self.exp["metric"]]
    ##            print "Score for "+query_type+" keywords=1:", baseline_score

                result={"query_type":query_type,
                        "fold":split_fold,
                        "score":baseline_score,
                        "method":method,
                        "type":"baseline",
                        "improvement":None,
                        "pct_improvement":None,
                        "num_data_points":len(retrieval_results)}
                for metric in metrics:
                    result[metric]=scores[0][metric]
                results.append(result)

                scores=self.measurePrecomputedResolution(test_set, method, keywords[query_type][method], query_type)
                this_score=scores[0][self.exp["metric"]]
    ##            print "Score with trained keywords:",this_score
                impro=this_score-baseline_score
                pct_impro=100*(impro/baseline_score) if baseline_score !=0 else 0
                improvements.append((impro*len(test_set))/len(retrieval_results))

                result={"query_type":query_type,
                        "fold":split_fold,
                        "score":this_score,
                        "method":method,
                        "type":"weight",
                        "improvement":impro,
                        "pct_improvement":pct_impro,
                        "num_data_points":len(retrieval_results)}
                if impro > 0:
                    better_zones.append(query_type)
                    better_zones_details.append((query_type,pct_impro))

                for metric in metrics:
                    result[metric]=scores[0][metric]
                for weight in keywords[query_type][method]:
                    result[weight]=keywords[query_type][method][weight]
                results.append(result)

            fold_result={"fold":split_fold,
                         "avg_improvement":sum(improvements)/float(len(improvements)) if len(improvements) > 0 else 0,
                         "num_improved_zones":len([x for x in improvements if x > 0]),
                         "num_zones":len(improvements),
                         "better_zones":better_zones,
                         "better_zones_details":better_zones_details,
                        }
            fold_results.append(fold_result)
            print("For fold",split_fold)
            print("Average improvement:",fold_result["avg_improvement"])
            print("keywords better than default in",fold_result["num_improved_zones"],"/",fold_result["num_zones"])
##            print("Better zones:",better_zones)
            print("Better zones, pct improvement:",better_zones_details)

        xtra="_".join(self.exp["train_keywords_for"])
        data=pd.DataFrame(results)
        data.to_csv(self.exp["exp_dir"]+self.exp["name"]+"_improvements_"+xtra+".csv")

        fold_data=pd.DataFrame(fold_results)
        fold_data.to_csv(self.exp["exp_dir"]+self.exp["name"]+"_folds_"+xtra+".csv")

    def measurePrecomputedResolution(self, retrieval_results, doc_method, keywords, citation_az="*"):
        """
            This is kind of like measureCitationResolution:
            it takes a list of precomputed retrieval_results, then applies the new
            parameters to them.

            All we need to do is adjust the keywords on the already available
            explanation formulas.

            :param keywords: dict
            :param retrieval_results: list of all result dicts
            :param doc_method: doc method we are measuring for

        """
        logger=ResultsLogger(False, dump_straight_to_disk=False) # init all the logging/counting
        logger.startCounting() # for timing the process, start now

        logger.setNumItems(len(retrieval_results),print_out=False)

        # for each query-result: (results are packed inside each query for each method)
        for result in retrieval_results:
            # select only the method we're testing for
            if "formulas" not in result:
                # there was an error reading this result
                continue

            formulas=result["formulas"]
            retrieved=runPrecomputedQuery(formulas,parameters)

            result_dict={"file_guid":result["file_guid"],
                         "citation_id":result["citation_id"],
                         "doc_position":result["doc_position"],
                         "query_method":result["query_method"],
                         "doc_method":doc_method,
                         "az":result["az"],
                         "cfc":result["cfc"],
                         "match_guid":result["match_guid"]}

            if not retrieved or len(retrieved)==0:    # the query was empty or something
##                print "Error: ", doc_method , qmethod,tfidfmodels[doc_method].indexDir
##                logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
                result_dict["mrr_score"]=0
                result_dict["precision_score"]=0
                result_dict["ndcg_score"]=0
                result_dict["rank"]=0
                result_dict["first_result"]=""

                logger.addResolutionResultDict(result_dict)
            else:
                result=logger.measureScoreAndLog(retrieved, result["citation_multi"], result_dict)

        logger.computeAverageScores()
        results=[]
        for query_method in logger.averages:
            for doc_method in logger.averages[query_method]:
                keywords=parameters
                data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

                for metric in logger.averages[query_method][doc_method]:
                    data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
                data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

                results.append(data_line)

        return results

    def trainKeywords(self):
        """
            Run the final stage of the training pipeline
        """
        gc.collect()
        options=self.options
        self.all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])

        trained_extractors={}
        if options.get("override_folds",None):
            self.exp["cross_validation_folds"]=options["override_folds"]

        if options.get("override_metric",None):
            self.exp["metric"]=options["override_metric"]

        numfolds=self.exp.get("cross_validation_folds",2)

        # First we train a keyword extractor from each fold's training set
        for split_fold in range(numfolds):
            print("\nFold #"+str(split_fold))
            trained_extractors[split_fold]=self.trainExtractor(split_fold)
            gc.collect()

        # Then we actually test them against the
        print("Now applying and testing keywords...\n")
        self.measureScoresOfKeywords(trained_extractors)




def main():
    logger=ResultsLogger(False,False)
##    logger.addResolutionResultDict
    pass

if __name__ == '__main__':
    main()
