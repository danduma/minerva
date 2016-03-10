# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

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
