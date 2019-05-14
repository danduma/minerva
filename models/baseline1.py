#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

import os
import json
from collections import Counter
from string import punctuation

# from sklearn import ensemble, svm, neural_network
from sklearn import metrics

from retrieval.elastic_functions import getElasticTermScores

from models.keyword_features import FeaturesReader, getTrainTestData, filterOutFeatures, flattenList, measurePR, \
    makeEvaluationQuery, tokenWeight, getRootDir
from models.base_model import BaseModel

from db.ez_connect import ez_connect
from tqdm import tqdm
import math

GLOBAL_FILE_COUNTER = 0


def listAllKeywordsToExtractFromReader(reader):
    """
        Lists all keywords that are marked as extract:true in a list or reader object
    """
    to_extract = []

    for kw_data in reader:
        if isinstance(kw_data, dict):
            best_kws = {t[0]: tokenWeight(t) for t in kw_data["best_kws"]}
            for sent in kw_data["context"]:
                for token in sent["token_features"]:
                    if token["text"].lower() in best_kws:
                        to_extract.append(token["text"].lower())
        elif isinstance(kw_data, tuple):
            if kw_data[1]:
                to_extract.append(kw_data[0]["text"])

    return Counter(to_extract)


def filterTermsFreq(terms, term_scores, min_docs_to_match, max_docs_to_match, min_term_len=0, max_docfreq=None):
    """
    filter terms that don't appear in a minimum of documents across the corpus
    """
    removed = {}
    res = []

    local_stopwords = ["__author", "__ref"]
    local_stopwords.extend(punctuation)

    for term in terms:
        to_remove = False

        freq = term_scores.get(term, {}).get("doc_freq", 0)
        if max_docfreq:
            assert freq <= max_docfreq

        if term in local_stopwords:
            to_remove = True
        elif max_docs_to_match and freq >= max_docs_to_match:
            to_remove = True
        elif min_docs_to_match and freq < min_docs_to_match:
            to_remove = True
        elif len(term) < min_term_len:
            to_remove = True
        if to_remove:
            removed[term] = removed.get(term, 0) + 1
        else:
            res.append(term)

    # print("Removed", removed)
    return res


class BaselineModel1(BaseModel):
    """
        Generate queries with all the terms but filtered
    """

    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        """
        """
        super(BaselineModel1, self).__init__(exp_dir, params, train_data_filename, test_data_filename)
        self.classifier = None
        self.use_weights = False
        self.filter_stopwords = params.get("filter_stopwords", True)

    def loadData(self):
        pass

    def postProcessLoadedData(self):
        pass

    def getKWsFromContext(self, context, context_tokens, ctx_index):
        """
        Return all terms as kws except some stopwords.
        Assign to each a weight analytically from the formulas.

        :param context:
        :param context_tokens:
        :param ctx_index:
        :return:
        """
        terms_to_extract = {t["text"].lower(): t["extract_weight"] for t in context["tokens"]}
        if self.filter_stopwords:
            terms_to_extract = {t: terms_to_extract[t] for t in terms_to_extract if
                                len(t) > 1 and t not in self.stopwords}

        predicted = []

        for token_text in context_tokens[ctx_index]:
            if token_text in terms_to_extract:
                if self.use_weights:
                    predicted.append(terms_to_extract[token_text])
                else:
                    predicted.append(True)
            else:
                if self.use_weights:
                    predicted.append(0)
                else:
                    predicted.append(False)
        return predicted

    def prepareTestData(self):
        pass

    def testModel(self, params={}):
        """
            Test the trained extractor on a test set
        """
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]
        if self.filter_stopwords:
            self.filterStopWordsInData(self.testing_contexts)

        context_tokens = []

        for context in self.testing_contexts:
            context_tokens.append([t["text"].lower() for t in context["tokens"]])

        self.X_test, self.y_test = getTrainTestData(self.testing_contexts, return_weight=self.use_weights)
        self.y_test = flattenList(self.y_test)

        self.prepareTestData()

        all_y_predicted = []
        result_contexts = []

        for ctx_index, context in enumerate(self.testing_contexts):
            for index, token in enumerate(context["tokens"]):
                context["tokens"][index]["extract_weight"] = context["weight_mask"][index]

            predicted = self.getKWsFromContext(context, context_tokens, ctx_index)
            if self.filter_stopwords:
                predicted = self.filterStopWordsInPredicted(context["tokens"], predicted)
                if len(self.extra_stopwords_removed) > 0:
                    print("Removed", len(self.extra_stopwords_removed), "tokens:", self.extra_stopwords_removed)

            assert len(predicted) == len(context_tokens[ctx_index])
            to_extract = []
            for index, extract in enumerate(predicted):
                if extract:
                    to_extract.append(context_tokens[ctx_index][index])
                else:
                    # print("Don't extract: ", context_tokens[ctx_index][index])
                    pass

            # for kw in context["best_kws"]:
            #     if kw[0] not in to_extract:
            #         print(kw[0], " missing")

            all_y_predicted.append(predicted)

            result_context = makeEvaluationQuery(context,
                                                 context_tokens[ctx_index],
                                                 predicted,
                                                 use_weight=self.use_weights)
            result_contexts.append(result_context)

        all_y_predicted = flattenList(all_y_predicted)
        if not self.use_weights:
            self.printClassificationResults(self.y_test, all_y_predicted)

        self.savePredictions(result_contexts)

    def printClassificationResults(self, y_test, predicted):
        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(y_test, predicted)))
        print("AUC:\n%f" % metrics.roc_auc_score(y_test, predicted, average="weighted"))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

    def informativeFeatures(self, features, fold=0):
        pass

    def defineModel(self):
        pass

    def trainModel(self):
        """
            Train an extractor
        """
        pass

    def trainSingleModel(self):
        pass


class BaselineModel2(BaselineModel1):
    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        """
        """
        super(BaselineModel2, self).__init__(exp_dir, params, train_data_filename, test_data_filename)
        self.corpus = ez_connect("AAC", "koko")
        self.index_name = "idx_az_ilc_az_annotated_aac_2010_1_paragraph"

    def getTermScoresFromElastic(self, term_scores_filename):
        for context in tqdm(self.testing_contexts, desc="Loading term vectors"):
            all_context_terms = [t["text"].lower() for t in context["tokens"]]
            all_context_terms = [t for t in all_context_terms if t not in self.term_scores]

            token_string = " ".join(all_context_terms)

            term_scores = getElasticTermScores(self.corpus.endpoint,
                                               token_string,
                                               self.index_name,
                                               "_all_text")

            term_scores = term_scores["_all_text"]["terms"]
            for term in term_scores:
                if term not in self.term_scores:
                    self.term_scores[term] = term_scores[term]
                    if term_scores[term].get("doc_freq", 0) > self.max_doc_freq:
                        self.max_doc_freq = term_scores[term]["doc_freq"]

        json.dump(self.term_scores, open(term_scores_filename, "w"))

    def prepareTestData(self):
        self.max_doc_freq = 0
        self.term_scores = {}

        term_scores_filename = os.path.join(self.exp_dir, "term_scores.json")
        if os.path.exists(term_scores_filename):
            self.term_scores = json.load(open(term_scores_filename, "r"))
            for term in self.term_scores:
                if self.term_scores[term].get("doc_freq", 0) > self.max_doc_freq:
                    self.max_doc_freq = self.term_scores[term]["doc_freq"]
        else:
            self.getTermScoresFromElastic(term_scores_filename)

        self.divisor = self.findIdealDocFreqThreshold()

        return

    def findIdealDocFreqThreshold(self):
        learning_rate = -0.1

        value = 2
        min_value = 00.7
        prev_score = 0
        iteration = 0

        all_scores = []

        while value > 0.1:
            all_f1 = []
            all_p = []
            all_r = []

            for context in self.testing_contexts:
                all_context_terms = [t["text"].lower() for t in context["tokens"]]

                # min_docs_to_match = max(5, int(self.max_doc_freq * 0.001))
                max_docs_to_match = self.max_doc_freq / value

                terms = filterTermsFreq(all_context_terms,
                                        self.term_scores,
                                        1,
                                        max_docs_to_match,
                                        max_docfreq=self.max_doc_freq)

                truth = set([kw[0] for kw in context["best_kws"]])
                p, r, tp, tp_fp, tp_fn = measurePR(truth, set(terms))
                division = (p + r)

                if division == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / division

                all_f1.append(f1)
                all_p.append(p)
                all_r.append(r)

            f1 = sum(all_f1) / len(all_f1)
            p = sum(all_p) / len(all_p)
            r = sum(all_r) / len(all_r)

            improvement = f1 - prev_score
            print("%d: F1 score: %0.4f Diff: %0.4f Divisor: %0.04f" % (iteration, f1, improvement, value))

            prev_score = f1

            all_scores.append((f1, value, p, r))
            value += learning_rate
            if value < min_value:
                break

        best_values = sorted(all_scores, key=lambda x: x[0], reverse=True)
        print("Best values: ")
        for val in best_values[:10]:
            print("%0.4f -> %0.4f F1 %0.04f Precision %0.04f Recall" % (val[1], val[0], val[2], val[3]))

        return best_values[0][1]

    def getKWsFromContext(self, context, context_tokens, ctx_index):
        all_context_terms = [t["text"].lower() for t in context["tokens"]]

        predicted = []

        min_docs_to_match = max(5, int(self.max_doc_freq * 0.001))

        max_docs_to_match = self.max_doc_freq / self.divisor

        terms = filterTermsFreq(all_context_terms,
                                self.term_scores,
                                1,
                                max_docs_to_match)

        weights = {}
        for term in terms:
            idf = 1 + math.log(self.max_doc_freq / self.term_scores[term]["doc_freq"])
            w = idf * idf

            weights[term] = w

        terms_to_extract = weights

        for token_text in context_tokens[ctx_index]:
            if token_text in terms_to_extract:
                if self.use_weights:
                    predicted.append(terms_to_extract[token_text])
                else:
                    predicted.append(True)
            else:
                if self.use_weights:
                    predicted.append(0)
                else:
                    predicted.append(False)
        return predicted


def main():
    model_class = BaselineModel1

    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    model = model_class(exp_dir,
                        train_data_filename="feature_data_at_w_min1.json.gz",
                        # test_data_filename="feature_data_test_at_w_1u1d.json.gz",
                        test_data_filename="feature_data_test_at_w.json.gz",
                        params={"filter_stopwords": False}
                        )

    # exp_dir = os.path.join(getRootDir("pmc_coresc"), "experiments", "pmc_generate_kw_trace")
    # model = model_class(exp_dir,
    #                     train_data_filename="feature_data_at_w_min1.json.gz",
    #                     test_data_filename="feature_data_test_at_w.json.gz",
    #                     params = {"filter_stopwords": False}
    #                     )

    model.use_weights = True
    model.run(external_test=True)

    pass


if __name__ == '__main__':
    main()
