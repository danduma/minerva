# Pipeline that uses retrieval results/formulas to annotate keywords to extract in citation contexts
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from collections import defaultdict
from math import log
import db.corpora as cp

from db.result_store import ElasticResultStorer, ResultDiskReader
from evaluation.precomputed_pipeline import PrecomputedPipeline
from evaluation.experiment import Experiment
from evaluation.query_generation import QueryGenerator

#
# DEPRECATED - Move to KeywordPipeline and KeywordTrainer
#

class ContextExtractor(QueryGenerator):
    """
        Modified query generator for finding the keywords
    """

    def __init__(self):
        pass


class ContextAnnotationPipeline(Experiment):
    """
        Runs like an experiment, at the point the weight training would. Goes
        over a subset of all PRRs.

        **OR**

        Does it use the PRRs or does it just get the explain() online?

        Needs the SciDocs augmented with all annotations first. Just like they get
        POS tags and AZ and CSC labels, each sentence has to get parsed. Other features
        like tfidf scores for each term I can learn later.

        parameters
            Phase 1: _full_text:1
            Phase 2: using fields

        Idea:
            Phase 1:

            For each "test" file, which for these purposes is a train file
                For each resolvable citation
                    Extract a block of text
                    Use whole block to generate query
                    Get results
                    Find keywords that will give best score
                    Annotate block of text with score for each word

            Phase 2:

            Annotate block of text with any and all features:

                POS
                Dependencies
                Document-wide TFIDF scores
                In_citation_sentence
                Coreference: Path_to_citation?
                Textual entailment? -- eventually

            What format to use for each sentence?
            Where to store all of this?
            Idea: instead of annotating each block individually, annotate whole paper with all the features?
                Not that crazy: can annotate all sentences and then

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

        # a commment
        term_weights=getFormulaTermWeights(unique_result)
        # TODO POS tagging, parsing, coreference resolution, etc.


    def runTestingPipeline(self):
        """
            This method overrides the standard experiment one, just does the
            annotation as it should.
        """
        if self.options.get("run_precompute_retrieval", False):
            pipeline=PrecomputedPipeline(retrieval_class=self.retrieval_class, use_celery=self.use_celery)
            pipeline.save_terms=True
            pipeline.runPipeline(self.exp)

    def run(self):
        """

        :return: None
        """
        self.selectTrainFiles()
        super(ContextAnnotationPipeline, self).run()

def main():

    from multi.celery_app import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")

    # train_set=
    pass

if __name__ == '__main__':
    main()
