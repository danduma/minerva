# FieldAgnostic/WeightTraining pipeline
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

import json, gc, random
from copy import deepcopy
from sklearn import cross_validation
import pandas as pd

import minerva.db.corpora as cp
from minerva.proc.results_logging import ProgressIndicator, ResultsLogger
##from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from minerva.proc.general_utils import getSafeFilename
from base_pipeline import getDictOfTestingMethods
from stored_formula import StoredFormula

GLOBAL_FILE_COUNTER=0


class weightCounterList(object):
    """
        Deals with iterating over combinations of weights for training
    """
    def __init__(self, values=[1,3,5]):
        """
            values = list of values each weight can take
        """
        self.values=values
        self.counters=[]
        self.MAX_COUNTER=len(values)-1

    def numCombinations(self):
        """
            Returns the number of total possible combinations of this counter
        """
        return pow(len(self.values),len(self.counters))

    def initWeights(self,parameters):
        """
            Creates counters and weights from input list of parameters to change
        """
        self.parameters=parameters
        # create a dict where every field gets a weight of x
        self.weights={x:self.values[0] for x in parameters}
        # make a counter for each field
        self.counters=[0 for _ in range(len(self.weights))]

    def allOnes(self):
        """
            Return all ones
        """
        return {x:1 for x in self.parameters}

    def allCombinationsProcessed(self):
        """
            Returns false if there is some counter that is not at its max
        """
        for counter in self.counters:
            if counter < self.MAX_COUNTER:
                return False
        return True

    def nextCombination(self):
        """
            Increases the counters and adjusts weights as necessary
        """
        self.counters[0]+=1

        for index in range(len(self.counters)):
            if self.counters[index] > self.MAX_COUNTER:
                self.counters[index+1] += 1
                self.counters[index]=0
            self.weights[self.weights.keys()[index]]=self.values[self.counters[index]]

    def getPossibleValues(self):
        return self.values


class WeightTrainer(object):
    """
        This class encapsulates all of the weight wrangling
    """
    def __init__(self, exp, options, use_celery=False):
        """
        """
        self.exp=exp
        self.options=options
        self.use_celery=use_celery
        self.all_doc_methods={}

    def dynamicWeightValues(self, split_fold):
        """
            Find the best combination of weights through dynamic programming, not
            testing every possible one, but selecting the best one at each stage
        """
        filename_add=""
        all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
        annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"] in ["annotated_boost"]]

        initialization_methods=[1]
    ##    initialization_methods=[1,"random"]
        MIN_WEIGHT=0
    ##    self.exp["movements"]=[-1,3]
        self.exp["movements"]=[-1,8,-2]

        best_weights={}

        numfolds=self.exp.get("cross_validation_folds",2)
    ##    counter=weightCounterList(exp["weight_values"])

        print("Processing zones ",self.exp["train_weights_for"])

        for zone_type in self.exp["train_weights_for"]:
            best_weights[zone_type]={}
            results=[]
            results_compare=[]

            retrieval_results=self.loadPrecomputedFormulas(zone_type)
            if len(retrieval_results) == 0:
                continue

            if len(retrieval_results) < numfolds:
                print("Number of results is smaller than number of folds for zone type ", zone_type)
                continue

            cv = cross_validation.KFold(len(retrieval_results), n_folds=numfolds, indices=True, shuffle=False, random_state=None, k=None)
            cv=[k for k in cv]

            traincv, testcv=cv[split_fold]
            train_set=[retrieval_results[i] for i in traincv]


            print("Training for citations in ",zone_type,"zones:",len(train_set),"/",len(retrieval_results))
            for method in annotated_boost_methods:
                res={}

                for weight_initalization in initialization_methods:
                    if weight_initalization==1:
    ##                    counter.initWeights(all_doc_methods[method]["runtime_parameters"])
                        weights={x:1 for x in all_doc_methods[method]["runtime_parameters"]}
                    elif weight_initalization=="random":
                        weights={x:random.randint(-10,10) for x in all_doc_methods[method]["runtime_parameters"]}
    ##                    counter.weights={x:random.randint(-10,10) for x in all_doc_methods[method]["runtime_parameters"]}

                    all_doc_methods[method]["runtime_parameters"]=weights

                    scores=self.measurePrecomputedResolution(train_set,method,weights, zone_type)

                    score_baseline=scores[0][self.exp["metric"]]
                    previous_score=score_baseline
                    first_baseline=score_baseline
                    score_progression=[score_baseline]

                    global GLOBAL_FILE_COUNTER
##                    drawWeights(self.exp,weights,zone_type+"_weights_"+str(GLOBAL_FILE_COUNTER))
##                    drawScoreProgression(self.exp,score_progression,zone_type+"_"+str(GLOBAL_FILE_COUNTER))
                    GLOBAL_FILE_COUNTER+=1

                    overall_improvement = score_baseline
                    passes=0

                    while passes < 3 or overall_improvement > 0:
                        for direction in self.exp["movements"]: # [-1,4,-2]
                            for index in range(len(weights)):
                                weight_name=weights.keys()[index]
                                prev_weight=weights[weight_name]
                                # hard limit of 0 for weights
                                weights[weight_name]=max(0,weights[weight_name]+direction)

                                scores=self.measurePrecomputedResolution(train_set,method,weights, zone_type)
                                this_score=scores[0][self.exp["metric"]]

                                if this_score <= previous_score:
                                    weights[weight_name]=prev_weight
                                else:
                                    previous_score=this_score

                        overall_improvement=this_score-score_baseline
                        score_baseline=this_score
                        score_progression.append(this_score)

                        # This is to export the graphs as weights are trained
##                        drawWeights(self.exp,weights,zone_type+"_weights_"+str(GLOBAL_FILE_COUNTER))
##                        drawScoreProgression(self.exp,{self.exp["metric"]:score_progression},zone_type+"_"+str(GLOBAL_FILE_COUNTER))
                        GLOBAL_FILE_COUNTER+=1

                        passes+=1

                    scores=self.measurePrecomputedResolution(train_set,method,weights, zone_type)
                    this_score=scores[0][self.exp["metric"]]

    ##                if split_fold is not None:
    ##                    split_set_str="_s"+str(split_fold)
    ##                else:
    ##                    split_set_str=""

    ##                print "Weight inialization:",weight_initalization
                    improvement=100*((this_score-first_baseline)/float(first_baseline)) if first_baseline > 0 else 0
                    print ("   Weights found, with score: {:.5f}".format(this_score)," Improvement: {:.2f}%".format(improvement))
                    best_weights[zone_type][method]=deepcopy(weights)
                    print ("   ",weights.values())

                    if self.exp.get("smooth_weights",None):
                        # this is to smooth a bit the weights in case they're too crazy
                        for weight in best_weights[zone_type][method]:
                            amount=abs(min(1,best_weights[zone_type][method][weight]) / float(3))
                            if best_weights[zone_type][method][weight] > 1:
                                best_weights[zone_type][method][weight] -= amount
                            elif best_weights[zone_type][method][weight] < 1:
                                best_weights[zone_type][method][weight] += amount

    ##                filename=self.exp["exp_dir"]+"weights_"+zone_type+"_[1, 3, 5]"+split_set_str+filename_add+".csv"
    ##                data=pandas.read_csv(filename,nrows=11)
                    res[weight_initalization]=this_score
    ##                print "Old weights:"
    ##                print data.iloc[0][CORESC_LIST+["avg_mrr","avg_precision"]]

                results_compare.append(res)

        better=0
        diff=0
    ##    for res in results_compare:
    ##        if res["random"] > res[1]:
    ##            better+=1
    ##        diff+=res[1]-res["random"]

    ##    print "Random inialization better than dynamic setting",better,"times"
    ##    print "Avg difference between methods:",diff/float(len(results_compare))
        for init_method in initialization_methods:
            if len(results_compare) > 0:
                avg=sum([res[init_method] for res in results_compare])/float(len(results_compare))
            else:
                avg=0
            print("Avg for ",init_method,":",avg)
    ##        if split_set is not None:
    ##            split_set_str="_s"+str(split_set)
    ##        else:
    ##            split_set_str=""
    ##        filename=getSafeFilename(self.exp["exp_dir"]+"weights_"+zone_type+"_"+str(counter.getPossibleValues())+split_set_str+filename_add+".csv")
    ##        data.to_csv(filename)

        return best_weights


    def loadPrecomputedFormulas(self, zone_type):
        """
            Loads the previously computed retrieval results, including query, etc.
        """
        return json.load(open(self.exp["exp_dir"]+"prr_"+self.exp["queries_classification"]+"_"+zone_type+".json","r"))


    def measureScores(self, best_weights):
        """
            Using precomputed weights from another split set, apply and report score
        """

        numfolds=self.exp.get("cross_validation_folds",2)

        results=[]
        fold_results=[]
        metrics=["avg_mrr","avg_ndcg", "avg_precision","precision_total"]

        print("Experiment:",self.exp["name"])
        print("Metric:",self.exp["metric"])
        print("Weight movements:",self.exp.get("movements",None))

        for split_fold in range(numfolds):
            weights=best_weights[split_fold]
            improvements=[]
            better_zones=[]

            for zone_type in self.exp["train_weights_for"]:
                retrieval_results=self.loadPrecomputedFormulas(zone_type)
                if len(retrieval_results) == 0:
                    continue

                if len(retrieval_results) < numfolds:
                    print("Number of results is smaller than number of folds for zone type ", zone_type)
                    continue

                cv = cross_validation.KFold(len(retrieval_results), n_folds=numfolds, indices=True, shuffle=False, random_state=None, k=None)
                cv=[k for k in cv]
                traincv, testcv=cv[split_fold]
        ##        train_set=retrieval_results[traincv[0]:traincv[-1]+1]
    ##            test_set=retrieval_results[testcv[0]:testcv[-1]+1]

                test_set=[ retrieval_results[i] for i in testcv ]

    ##            print (testcv[0],testcv[-1])

    ##            if split_fold==0:
    ##                test_set=retrieval_results[:len(retrieval_results)/2]
    ##                print (0,len(retrieval_results)/2)
    ##            elif split_fold==1:
    ##                print (len(retrieval_results)/2,len(retrieval_results))
    ##                test_set=retrieval_results[len(retrieval_results)/2:]

                for method in weights[zone_type]:
                    weights_baseline={x:1 for x in self.all_doc_methods[method]["runtime_parameters"]}
                    scores=self.measurePrecomputedResolution(test_set, method, weights_baseline, zone_type)
                    baseline_score=scores[0][self.exp["metric"]]
        ##            print "Score for "+zone_type+" weights=1:", baseline_score
                    result={"zone_type":zone_type,
                            "fold":split_fold,
                            "score":baseline_score,
                            "method":method,
                            "type":"baseline",
                            "improvement":None,
                            "pct_improvement":None,
                            "num_data_points":len(retrieval_results)}
                    for metric in metrics:
                        result[metric]=scores[0][metric]
                    for weight in weights[zone_type][method]:
                        result[weight]=1
                    results.append(result)

                    scores=self.measurePrecomputedResolution(test_set, method, weights[zone_type][method], zone_type)
                    this_score=scores[0][self.exp["metric"]]
        ##            print "Score with trained weights:",this_score
                    impro=this_score-baseline_score
                    pct_impro=100*(impro/baseline_score) if baseline_score !=0 else 0
                    if impro > 0:
                        better_zones.append(zone_type)
                    improvements.append((impro*len(test_set))/len(retrieval_results))

                    result={"zone_type":zone_type,
                            "fold":split_fold,
                            "score":this_score,
                            "method":method,
                            "type":"weight",
                            "improvement":impro,
                            "pct_improvement":pct_impro,
                            "num_data_points":len(retrieval_results)}
                    for metric in metrics:
                        result[metric]=scores[0][metric]
                    for weight in weights[zone_type][method]:
                        result[weight]=weights[zone_type][method][weight]
                    results.append(result)

            fold_result={"fold":split_fold,
                         "avg_improvement":sum(improvements)/float(len(improvements)),
                         "num_improved_zones":len([x for x in improvements if x > 0]),
                         "num_zones":len(improvements),
                         "better_zones":better_zones,
                        }
            fold_results.append(fold_result)
            print("For fold",split_fold)
            print("Average improvement:",fold_result["avg_improvement"])
            print("Weights better than default in",fold_result["num_improved_zones"],"/",fold_result["num_zones"])
            print("Better zones:",better_zones)


        data=pd.DataFrame(results)
        data.to_csv(self.exp["exp_dir"]+self.exp["name"]+"_improvements.csv")

        fold_data=pd.DataFrame(fold_results)
        fold_data.to_csv(self.exp["exp_dir"]+self.exp["name"]+"_folds.csv")

##        print("Avg % improvement per zone:")
##        means=data[["zone_type","pct_improvement"]].groupby("zone_type").mean().sort(columns=["pct_improvement"],ascending=False)
##        means=means.join(data[["zone_type","pct_improvement"]].groupby("zone_type").std())
##        print(means)

    def measureCitationResolution(self, files_dict, precomputed_queries, all_doc_methods,
    citation_az, testing_methods, retrieval_class=None, full_corpus=False):
        """
            Use Citation Resolution to measure the impact of different runtime parameters.

            files_dict is a dict where each key is a guid:
                "j97-3003": {"tfidf_models": [{"actual_dir": "C:\\NLP\\PhD\\bob\\fileDB\\LuceneIndeces\\j97-3003\\ilc_az_annotated_1_20",
    			"method": "az_annotated_1_ALL"}],
    		"resolvable_citations": 16,
    		"doc_methods": {"az_annotated_1_ALL": {
    				"index": "ilc_az_annotated_1_20",
    				"runtime_parameters": ["AIM",
    				"BAS",
    				"BKG",
    				"CTR",
    				"OTH",
    				"OWN",
    				"TXT"],
    				"parameters": [1],
    				"type": "annotated_boost",
    				"index_filename": "ilc_az_annotated_1_20",
    				"parameter": 1,
    				"method": "az_annotated"
    			}
    		},
    		"guid": "j97-3003",
    		"in_collection_references": 10}
        """
        logger=ResultsLogger(False) # init all the logging/counting
        logger.startCounting() # for timing the process, start now

        logger.setNumItems(len(files_dict),print_out=False)

        tfidfmodels={}

        # if we're running over the full cp.Corpus, we should only load the indices once
        # at the beginning, as they're going to be the same
        if full_corpus:
            for model in files_dict["ALL_FILES"]["tfidf_models"]:
                # create a Lucene search instance for each method
                tfidfmodels[model["method"]]=retrieval_class(model["actual_dir"],model["method"],logger=None)

        previous_guid=""
        logger.total_citations=len(precomputed_queries)

        #===================================
        # MAIN LOOP over all testing files
        #===================================
        for query in precomputed_queries:
            # for every generated query for this context
            guid=query["file_guid"]

            # if this is not a full_corpus run and we've moved on to the next test file
            # then we should load the indices for this new file
            if not full_corpus and guid != previous_guid:
    ##            logger.showProgressReport(guid) # prints out info on how it's going
                previous_guid=guid
                for model in files_dict[guid]["tfidf_models"]:
                    # create a Lucene search instance for each method
                    tfidfmodels[model["method"]]=retrieval_class(model["actual_dir"],model["method"],logger=None)

            # for every method used for extracting BOWs
            for doc_method in all_doc_methods:
                # ACTUAL RETRIEVAL HAPPENING - run query
                retrieved=tfidfmodels[doc_method].runQuery(query["query_text"],all_doc_methods[doc_method]["runtime_parameters"], guid)

                result_dict={"file_guid":guid,
                             "citation_id":query["citation_id"],
                             "doc_position":query["doc_position"],
                             "query_method":query["query_method"],
                             "doc_method":doc_method ,
                             "az":query["az"],
                             "cfc":query["cfc"],
                             "match_guid":query["match_guid"]}

                if not retrieved:    # the query was empty or something
    ##                        print "Error: ", doc_method , qmethod,tfidfmodels[method].indexDir
    ##                        logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
                    result_dict["mrr_score"]=0
                    result_dict["precision_score"]=0
                    result_dict["ndcg_score"]=0
                    result_dict["rank"]=0
                    result_dict["first_result"]=""

                    logger.addResolutionResultDict(result_dict)
                else:
                    logger.measureScoreAndLog(retrieved, query["citation_multi"], result_dict)


        logger.computeAverageScores()
        results=[]
        for query_method in logger.averages:
            for doc_method in logger.averages[query_method]:
                weights=all_doc_methods[doc_method]["runtime_parameters"]
                data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

                for metric in logger.averages[query_method][doc_method]:
                    data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
                data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

                signature=""
                for w in weights:
                    data_line[w]=weights[w]
                    signature+=str(w)

                data_line["weight_signature"]=signature
                results.append(data_line)

    ##    logger.writeDataToCSV(cp.Corpus.dir_output+"testing_test_precision.csv")

        return results


    def runPrecomputedQuery(self, retrieval_result, parameters):
        """
            This takes a query that has already had the results added
        """
        scores=[]
        for unique_result in retrieval_result:
            formula=StoredFormula(unique_result["formula"])
            score=formula.computeScore(formula.formula, parameters)
            scores.append((score,{"guid":unique_result["guid"]}))

        scores.sort(key=lambda x:x[0],reverse=True)
        return scores


    def measurePrecomputedResolution(self, retrieval_results,method,parameters, citation_az="*"):
        """
            This is kind of like measureCitationResolution:
            it takes a list of precomputed retrieval_results, then applies the new
            parameters to them. This is how we recompute what Lucene gives us,
            avoiding having to call Lucene again.

            All we need to do is adjust the weights on the already available
            explanation formulas.
        """
        logger=ResultsLogger(False, dump_straight_to_disk=False) # init all the logging/counting
        logger.startCounting() # for timing the process, start now

        logger.setNumItems(len(retrieval_results),print_out=False)

        # for each query-result: (results are packed inside each query for each method)
        for result in retrieval_results:
            # select only the method we're testing for
            formulas=result["formulas"]
            retrieved=self.runPrecomputedQuery(formulas,parameters)

            result_dict={"file_guid":result["file_guid"],
                         "citation_id":result["citation_id"],
                         "doc_position":result["doc_position"],
                         "query_method":result["query_method"],
                         "doc_method":method,
                         "az":result["az"],
                         "cfc":result["cfc"],
                         "match_guid":result["match_guid"]}

            if not retrieved or len(retrieved)==0:    # the query was empty or something
    ##                        print "Error: ", doc_method , qmethod,tfidfmodels[method].indexDir
    ##                        logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
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
    ##            weights=all_doc_methods[doc_method]["runtime_parameters"]
                weights=parameters
                data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

                for metric in logger.averages[query_method][doc_method]:
                    data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
                data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

##                signature=""
##                for w in weights:
##                    data_line[w]=weights[w]
##                    signature+=str(w)
    ##            data_line["weight_signature"]=signature
                results.append(data_line)

    ##    logger.writeDataToCSV(cp.cp.Corpus.dir_output+"testing_test_precision.csv")

        return results

    def trainWeights(self):
        """
            Run the final stage of the weight training pipeline.
        """
        gc.collect()
        options=self.options
        self.all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])

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
            best_weights[split_fold]=self.dynamicWeightValues(split_fold)

        print("Now applying and testing weights...\n")
        self.measureScores(best_weights)


def autoTrainWeightValues(zones_to_process=[], files_dict_filename="files_dict.json", testing_methods=None, filename_add=""):
    """
        Tries different values for different zones given the zone of the
        sentence where the citation occurs.
    """
    files_dict=json.load(open(cp.Corpus.paths.prebuiltBOWs+files_dict_filename,"r"))
    queries_by_az=json.load(open(cp.Corpus.paths.prebuiltBOWs+"queries_by_az.json","r"))
    queries_by_cfc=json.load(open(cp.Corpus.paths.prebuiltBOWs+"queries_by_cfc.json","r"))

    print("AZs:")
    for key in queries_by_az:
        print(key, len(queries_by_az[key]))

    print ("\nCFCs:")
    for key in queries_by_cfc:
        print(key, len(queries_by_cfc[key]))
    print ("\n")

    results=[]

    all_doc_methods=getDictOfTestingMethods(testing_methods)

    annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"]=="annotated_boost"]

    counter=weightCounterList([1,3,5])
    print(counter.getPossibleValues())
    # this is to run it "in polynomial time"
    # most citations are either OTH (1307) or OWN (3876)
    MAX_QUERIES_EVALUATED=650

    print("Processing zones ",zones_to_process)

    for az_type in zones_to_process:
        print("Number of queries in ",az_type,"zones:",len(queries_by_az[az_type]))
        if len(queries_by_az[az_type]) > MAX_QUERIES_EVALUATED:
            print("Evaluating a maximum of", MAX_QUERIES_EVALUATED)
        for method in annotated_boost_methods:

            progress=ProgressIndicator(True) # create a progress indicator, keeps track of what's been done and how long is left
            counter.initWeights(all_doc_methods[method]["runtime_parameters"])
            all_doc_methods[method]["runtime_parameters"]=counter.weights

            # total number of combinations that'll be processed
            progress.setNumItems(counter.numCombinations())

            while not counter.allCombinationsProcessed():#  and progress.item_counter < 2:

                weight_parameter={method:{"runtime_parameters":counter.weights}}
##                print counters
                # !TODO set :MAX_QUERIES_EVALUATED and MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2
##                citations_set=queries_by_az[az_type][MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2]
                citations_set=queries_by_az[az_type][:MAX_QUERIES_EVALUATED]
                scores=measureCitationResolution(files_dict,citations_set, weight_parameter, az_type, all_doc_methods)
                results.extend(scores)
                progress.showProgressReport("",1)
                counter.nextCombination()

        # prints results per each AZ/CFC zone
        data=pd.DataFrame(results)
        metric="avg_mrr"
        data=data.sort(metric, ascending=False)
        filename=getSafeFilename(cp.Corpus.paths.output+"weights_"+az_type+"_"+str(counter.getPossibleValues())+filename_add+".csv")
        data.to_csv(filename)
##        statsOnResults(data, metric)

##        print data.to_string()

def autoTrainWeightValues_optimized(exp, split_set=None):
    """
        Tries different values for
    """
    filename_add=""
    all_doc_methods=getDictOfTestingMethods(exp["doc_methods"])

    annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"]=="annotated_boost"]

    counter=weightCounterList(exp["weight_values"])
    print(counter.getPossibleValues())

    print("Processing zones ",exp["train_weights_for"])

    for zone_type in exp["train_weights_for"]:
        results=[]
        retrieval_results=json.load(open(exp["exp_dir"]+"prr_"+exp["queries_classification"]+"_"+zone_type+".json","r"))
        print("Number of precomputed results in ",zone_type,"zones:",len(retrieval_results))
        for method in annotated_boost_methods:
            counter.initWeights(all_doc_methods[method]["runtime_parameters"])

            progress=ProgressIndicator(True) # create a progress indicator, keeps track of what's been done and how long is left
            all_doc_methods[method]["runtime_parameters"]=counter.weights

            # total number of combinations that'll be processed
            progress.setNumItems(counter.numCombinations(),dot_every_xitems=max(10,80-(len(retrieval_results)/40)))

            print("Testing weight value combinations with precomputed formula")
            while not counter.allCombinationsProcessed():#  and progress.item_counter < 2:
                if split_set==1:
                    test_set=retrieval_results[:len(retrieval_results)/2]
                elif split_set==2:
                    test_set=retrieval_results[len(retrieval_results)/2:]
                elif not split_set:
                    test_set=retrieval_results
                else:
                    assert(split_set in [None,1,2])

                scores=measurePrecomputedResolution(test_set,method,counter.weights, zone_type)

                results.extend(scores)
                progress.showProgressReport("",1)
                counter.nextCombination()

        # prints results per each AZ/CFC zone
        data=pd.DataFrame(results)
        metric="avg_mrr"
        data=data.sort(metric, ascending=False)

        if split_set is not None:
            split_set_str="_s"+str(split_set)
        else:
            split_set_str=""
        filename=getSafeFilename(exp["exp_dir"]+"weights_"+zone_type+"_"+str(counter.getPossibleValues())+split_set_str+filename_add+".csv")
        data.to_csv(filename)


def main():
    pass

if __name__ == '__main__':
    main()
