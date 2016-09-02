# Pipeline that uses the precomputed retrieval results/formulas to annotate citation contexts
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from collections import defaultdict
from math import log

from minerva.evaluation.precomputed_pipeline import PrecomputedPipeline
from minerva.evaluation.experiment import Experiment
from minerva.evaluation.query_generation import QueryGenerator

def termScoresInFormula(part):
    """
        Returnslist of all term matching elements in formula

        :param part: tuple, list or dict
        :returns: list of all term matching elements in formula
    """
    if isinstance(part, tuple) or isinstance(part, list):
        return part
    elif isinstance(part, dict):
        scores=[termScoreInFormula(sub_part, parameters) for sub_part in part["parts"]]
        result=[]
        for score in scores:
            if isinstance(score, list):
                result.extend(score)
            else:
                result.append(score)
        return result

def getDictOfTermScores(formula):
    """
        Returns the score of each term in the formula
    """
    term_scores=termScoresInFormula(formula)
    res={}
    for score in term_scores:
        # score=(field,qw,fw,term)
        res[score[3]]=res.get(score[3],0)+(score[1]*score[2])
    return res


def getFormulaTermWeights(unique_result):
    """
        Computes a score for each matching keyword in the formula for the
        matching
    """
    idf_scores=defaultdict(0)
    max_scores=defaultdict(0)

    formula_term_scores=[]
    match_result=None

    for formula in unique_result["formulas"]:
        term_scores=getDictOfTermScores(formula["formula"])
        formula_term_scores.append((formula,term_scores))
        if formula["guid"] == unique_result["match_guid"]:
            match_result=term_scores

        for term in term_scores:
            idf_scores[term]=idf_scores[term]+term_scores[term]
            if term_scores[term] > max_scores[term]:
                max_scores[term] = term_scores[term]

    if not match_result:
        return None

    for term in idf_scores:
        idf_scores[term] = log((max_scores[term] * len(unique_result["formulas"])) / (1 + idf_scores[term]), 2)

    for term in match_result:
        match_result[term] = match_result[term] * idf_scores[term]

    return match_result

class ContextExtractor(QueryGenerator):
    """
        Modified query generator for context
    """

    def __init__():



class ContextAnnotationPipeline(Experiment):
    """
    """

    def bindAllExtractors(self):
        """
        """
        self.query_generator.options["add_full_sentences"]=True
        super(self.__class__, self).bindAllExtractors()

    def loadPrecomputedFormulas(self, zone_type):
        """
            Loads the previously computed retrieval results, including query, etc.
        """
        prr=ElasticResultStorer(self.exp["name"],"prr_"+self.exp["queries_classification"]+"_"+zone_type, cp.Corpus.endpoint)
        reader=ResultDiskReader(prr, cache_dir=os.path.join(self.exp["exp_dir"], "cache"), max_results=700)
        reader.bufsize=30
        return reader

##        return prr.readResults(250)
##        return json.load(open(self.exp["exp_dir"]+"prr_"+self.exp["queries_classification"]+"_"+zone_type+".json","r"))


    def annotateContext(self, unique_result):
        """
            This method returns the fully annotated context, with term weights
            and all necesary features
        """
        terms={}

        term_weights=getFormulaTermWeights(unique_result)
        # TODO POS tagging, parsing, coreference resolution, etc.


    def runTestingPipeline(self):
        """
        """
        if self.options.get("run_precompute_retrieval", False):
            pipeline=PrecomputedPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
            pipeline.save_terms=True
            pipeline.runPipeline(self.exp)

        self.all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])

        if options.get("override_folds",None):
            self.exp["cross_validation_folds"]=options["override_folds"]

        if options.get("override_metric",None):
            self.exp["metric"]=options["override_metric"]


def main():
    pass

if __name__ == '__main__':
    main()
