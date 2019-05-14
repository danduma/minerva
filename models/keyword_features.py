# Functions to load, save and convert token features for ML training


import gzip
import json
import os
import re
import sys
from copy import deepcopy
from collections import Counter

from matplotlib import pyplot as plt
from six import string_types

import numpy as np

np.warnings.filterwarnings('ignore')

import pandas as pd

from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST

SENTENCE_FEATURES_TO_COPY = ["az", "csc_type"]

SENTENCE_END_TOKEN = {
    "text": "#STOP#",
    "pos": 96,
    "pos_": "PUNCT",
    "dep": 442,
    "dep_": "punct",
    "ent_type": 0,  # this is useless for scientific papers. Need a corpus-specific NER
    "lemma_": ".",
    "lemma": 5,  # "IS_PUNCT"
    "is_punct": True,
    "is_lower": False,
    "like_num": False,  # Note this will catch 0, million, two, 1,938 and others
    "is_stop": False,
    "tf": 0,
    "df": 0,
    "tf_idf": 0,
    "token_type": "p",  # "t" for normal token, "c" for citation, "p" for punctuation
    "dist_cit": 0
}

PAD_TOKEN = {
    "text": "#PAD#",
    "pos": 96,
    "pos_": "PUNCT",
    "dep": 442,
    "dep_": "punct",
    "ent_type": 0,
    "lemma_": ".",
    "lemma": 0,
    "is_punct": True,
    "is_lower": False,
    "like_num": False,  # Note this will catch 0, million, two, 1,938 and others
    "is_stop": False,
    "tf": 0,
    "df": 0,
    "tf_idf": 0,
    "token_type": "p",  # "t" for normal token, "c" for citation, "p" for punctuation
    "dist_cit": 0
}

token_type_to_int = {"t": 0, "c": 1, "p": 2}
az_to_int = {"": 0}
coresc_to_int = {"": 0}

for index, az in enumerate(AZ_ZONES_LIST):
    az_to_int[az] = index

for index, csc in enumerate(CORESC_LIST):
    coresc_to_int[csc] = index


class FeaturesReader(object):
    """
    Reads sequentially from a text or gzipped file, where each line
    is a json object

    """

    def __init__(self, path, max_items=sys.maxsize):
        self.position = 0
        self.numlines = None
        self.max_items = max_items
        if path.endswith(".gz"):
            self.f = gzip.open(path, "rt")
        else:
            self.f = open(path, "r")

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= self.max_items:
            raise StopIteration

        line = self.f.readline()
        if line is None or line == "":
            raise StopIteration

        self.position += 1

        return self.loadLine(line)

    def next(self):
        return self.__next__()

    def __len__(self):
        if self.numlines:
            return self.numlines
        else:
            cur_pos = self.f.tell()
            self.numlines = sum(1 for line in self.f)
            self.numlines = min(self.numlines, self.max_items)
            self.f.seek(cur_pos)

            return self.numlines

    def seek(self, offset, whence):
        self.f.seek(offset, whence)

    def loadLine(self, line):
        return json.loads(line.strip("\n"))

    def getIterator(self, size=sys.maxsize):
        max_available = self.max_items - self.position
        size = min(max_available, size)
        for count in range(size):
            line = self.f.readline()
            if line is None or line == "":
                break

            yield self.loadLine(line)

    def getMiniBatch(self, size=10):
        res = [line for line in self.getIterator(size)]

        return res

    def readAll(self):
        return self.getMiniBatch(size=sys.maxsize)


# =================


def getAnnotationListsForContext(all_token_features, all_keywords):
    """
        Returns a list of (token_features_dict, {True,False}, weight) tuples.
        For each token, based on its features, the token is annotated as
        to-be-extracted or not
    """
    res = []

    all_keys = list(all_keywords.keys())

    for token in all_token_features:
        token_text = token["text"].lower()
        extract = (token_text in all_keys)
        weight = token.get("weight", all_keywords.get(token_text, 0.0))
        if weight is None:
            weight = 0.0
        try:
            del token["weight"]
            del token["extract"]
        except:
            pass
        res.append((token, extract, weight))
    return res


def getOneContextFeatures(context):
    """
        Prepares a single context's data for any nn. Takes ["token_features"] from list
        of sentences and returns a single list of token features.
    """
    context_tokens = []

    all_keywords = {t[0]: tokenWeight(t) for t in context["best_kws"]}
    for sentence in context["context"]:
        for feature in SENTENCE_FEATURES_TO_COPY:
            for token_feature in sentence["token_features"]:
                token_feature[feature] = sentence.get(feature, "")
                assert token_feature["pos"] != 0

        ##            for token_feature in sentence["token_features"]:
        ##                for key in token_feature:
        ##                    if key.startswith("dist_cit_"):
        sent_tokens = deepcopy(sentence["token_features"])
        sent_tokens.append(SENTENCE_END_TOKEN)
        context_tokens.extend(sent_tokens)

    featureset = getAnnotationListsForContext(context_tokens, all_keywords)
    tokens, to_extract, weights = zip(*featureset)

    # print(json.dumps(context))
    # assert False
    context_features = {
        "file_guid": context["file_guid"],
        "cit_ids": context["cit_ids"],
        "citation_id": context["cit_ids"][0],
        "match_guids": context["match_guids"],
        "vis_text": context.get("vis_text", ""),
        "query_text": context.get("query_text", ""),
        "keyword_selection_entry": context.get("keyword_selection_entry", ""),
        "kw_selection_scores": context["kw_selection_scores"],
        "kw_selection_weight_scores": context["kw_selection_weight_scores"],
        "original_scores": {
            "ndcg_score": context["ndcg_score"],
            "precision_score": context["precision_score"],
            "mrr_score": context["mrr_score"],
        },
        "per_guid_rank": context["per_guid_rank"],
        "citation_multi": context["cit_multi"],

        "extract_mask": to_extract,
        "tokens": tokens,
        "weight_mask": weights,
        "best_kws": context["best_kws"],
    }

    return context_features


def saveFeatureData(precomputed_contexts, path):
    """
        Calls getContextFeatures() and dumps prepared data to json file
    """
    if path.endswith(".gz"):
        f = gzip.open(path, "wt")
    else:
        f = open(path, "w")

    written = 0
    print("Exporting feature data...")
    for context in precomputed_contexts:
        feature_data = getOneContextFeatures(context)

        f.write(json.dumps(feature_data) + "\n")
        written += 1

    f.close()
    print("Total written", written)


def loadFeatureData(path):
    lines = []
    if path.endswith(".gz"):
        f = gzip.open(path, "rt")
    else:
        f = open(path, "r")

    for line in f:
        lines.append(json.loads(line.strip("\n")))

    f.close()
    return lines


def filterFeatures(features, ignore_features):
    """
    Returns a dict lacking the features in ignore_features

    """
    if len(ignore_features) == 0:
        return features

    new_dict = {}
    for feature in [f for f in features if f not in set(ignore_features)]:
        new_dict[feature] = features[feature]
    return new_dict


def getTokenListFromContexts(contexts):
    """
    Returns a single list of tokens form all of the contexts in feature_data

    :param contexts: list of contexts, each of which is a list of tokens, each of which a dict of features
    :return: list
    """
    all_tokens = []
    if len(contexts) > 0 and contexts[0][0][0]["text"].lower() != "#SENT_END#":
        all_tokens.append((SENTENCE_END_TOKEN, False, 0))

    for context in contexts:
        all_tokens.extend(context)
    return all_tokens


def padTokens(token_list, set_len):
    """
    Adds PAD_TOKEN at the end of token_list to make it the specified length

    :param token_list: list of tokens
    :param set_len: desired length
    :return: padded list
    """
    if len(token_list) >= set_len:
        return token_list[:set_len]

    res = list(token_list)

    tokens_to_add = set_len - len(token_list)

    for counter in range(tokens_to_add):
        res.append((PAD_TOKEN, False, 0))

    return res


def unPadTokens(token_list):
    """
    Remove padding tokens from list

    :param token_list:
    :return:
    """
    if len(token_list) == 0:
        return token_list

    res = []

    for element in token_list:
        if element[0]["text"] != PAD_TOKEN["text"]:
            res.append(element)

    return res


def getMatrixFromTokenList(contexts):
    """
    Returns a numpy matrix with all the data, ready to train a model

    :param token_list:
    :return:
    """
    NUM_FEATURES = 14 + 2  # adding to_extract, weight

    max_len = max([len(context) for context in contexts])

    matrix = np.zeros((len(contexts), max_len, NUM_FEATURES), dtype=np.float32)

    for cnt, context in enumerate(contexts):
        token_list = padTokens(context, max_len)
        for i, packed_token in enumerate(token_list):
            token, extract, weight = packed_token

            lemma = token.get("lemma", 0)
            if isinstance(lemma, string_types):
                matrix[cnt, i, 0] = 0  # the Spacy lemma is the int value of the unique token from the tokenizer
            else:
                matrix[cnt, i, 0] = lemma

            matrix[cnt, i, 1] = token.get("pos", 0)
            matrix[cnt, i, 2] = token.get("dep", 0)
            matrix[cnt, i, 3] = int(token.get("is_lower", 0))
            matrix[cnt, i, 4] = int(token.get("is_punct", 0))
            matrix[cnt, i, 5] = int(token.get("like_num", 0))
            matrix[cnt, i, 6] = int(token.get("is_stop", 0))
            matrix[cnt, i, 7] = token.get("tf", 0)
            matrix[cnt, i, 8] = token.get("df", 0)
            matrix[cnt, i, 9] = token.get("tf_idf", 0)
            matrix[cnt, i, 10] = az_to_int[token.get("az", "")]
            matrix[cnt, i, 11] = coresc_to_int[token.get("csc_type", "")]
            matrix[cnt, i, 12] = token.get("dist_cit", 0)
            matrix[cnt, i, 13] = token_type_to_int[token.get("token_type", "t")]

            matrix[cnt, i, 14] = int(extract)
            matrix[cnt, i, 15] = weight
    return matrix


def normaliseWeights(contexts):
    all_scores = []
    for context in contexts:
        all_scores.extend(context["weight_mask"])

    dmax = max(all_scores)
    dmin = min(all_scores)
    diff = float(dmax - dmin)

    for context in contexts:
        new_weights = [w - dmin / diff for w in context["weight_mask"]]
        context["weight_mask"] = new_weights

    return dmin, diff


def normaliseFeatures(contexts,
                      features_to_normalise=["dist_cit", "tf", "tf_idf"]):
    # from tqdm import tqdm
    max_vals = {}
    min_vals = {}
    for context in contexts:
        for index, features in enumerate(context["tokens"]):
            if features["text"] in ["#STOP#", "#PAD#"]:
                continue

            for norm_feat in features_to_normalise:
                if norm_feat not in max_vals:
                    max_vals[norm_feat] = features[norm_feat]
                    min_vals[norm_feat] = features[norm_feat]
                else:
                    if features[norm_feat] > max_vals[norm_feat]:
                        max_vals[norm_feat] = features[norm_feat]
                    if features[norm_feat] < min_vals[norm_feat]:
                        min_vals[norm_feat] = features[norm_feat]

    for context in contexts:
        for features in context["tokens"]:
            # print(features["text"])

            for norm_feat in features_to_normalise:
                if norm_feat in max_vals:
                    if norm_feat == "dist_cit" and features["text"] in ["#STOP#", "#PAD#"]:
                        continue

                    # print(norm_feat)
                    # print(features[norm_feat])
                    if features[norm_feat] == 0:
                        continue
                    # features[norm_feat] = (features[norm_feat] - min_vals[norm_feat]) / float(
                    #     max_vals[norm_feat] - min_vals[norm_feat])
                    features[norm_feat] = features[norm_feat] / float(max_vals[norm_feat])

                    # print(features[norm_feat])
                    # print()

    return contexts


def filterOutFeatures(context, features_to_remove, corpus):
    """
    :param contexts:
    :param features_to_remove:
    :return:
    """
    context = deepcopy(context)
    for index, features in enumerate(context["tokens"]):
        # if "az" in features and "csc_type" in features:
        #     print("hmm")

        if corpus == "aac" and "csc_type" in features:
            del features["csc_type"]
        if corpus == "pmc" and "az" in features:
            del features["az"]

        # assert not ("az" in features and "csc_type" in features)
        # remove the ints equivalent to the strings
        for to_delete in features_to_remove:
            if to_delete in features:
                del features[to_delete]

    return context


def getTrainTestData(contexts, return_weight=False):
    """
    Split contexts to X and y datasets, filter out features we don't want, add some we do.

    :param tagged_sentences: a list of POS tagged sentences
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return:
    """
    X, y = [], []

    for context in contexts:
        X.append(context["tokens"])

        if return_weight:
            y.append(context["weight_mask"])
        else:
            y.append(context["extract_mask"])

    return X, y


def saveMatrix(filename, matrix):
    """
    Saves the numpy matrix compressed in gzip

    :param filename: file to save to
    :param matrix: the matrix to save
    :return: None
    """
    f = gzip.GzipFile(filename, "w")
    np.save(file=f, arr=matrix, fix_imports=True)
    f.close()


def loadMatrix(filename):
    """
    Loads the compressed numpy matrix

    :param filename: file to load from
    :return: None
    """
    f = gzip.GzipFile(filename, "r")
    return np.load(file=f)


def makeEvaluationQuery(context, context_tokens, predicted, use_weight=False):
    """
    Exports a precomputed query to just evaluate

    :param context:
    :param predicted:
    :return:
    """
    result_context = deepcopy(context)
    for to_del in ["tokens", "extract_mask", "weight_mask"]:
        if to_del in result_context:
            del result_context[to_del]

    counts = Counter([t.lower() for t in context_tokens])

    query = []
    for index, extract in enumerate(predicted):
        if extract > 0:
            text = context_tokens[index]
            if use_weight:
                weight = extract
            else:
                weight = 1
            # (text, count, boost, bool, field, distance)
            # new_token = {"token": text, "count": counts[text], "boost": float(weight)}
            # new_token = (text, counts[text], float(weight)/counts[text], None, None, None)
            new_token = (text, counts[text], float(weight), None, None, None)
            query.append(new_token)

    # result_context["structured_query"] = StructuredQuery(query)
    result_context["structured_query"] = query
    return result_context


def convert_json_to_matrix(features_filename, features_numpy_filename):
    feature_data = loadFeatureData(features_filename)
    matrix = getMatrixFromTokenList(feature_data)
    saveMatrix(features_numpy_filename, matrix)


def convert_json_to_csv(features_filename, features_csv_filename):
    feature_data = loadFeatureData(features_filename)

    all_tokens = []

    for cnt, context in enumerate(feature_data):
        # token_list = padTokens(context, max_len)
        token_list = context
        for i, packed_token in enumerate(token_list):
            token, extract, weight = packed_token
            new_token = deepcopy(token)
            new_token["extract"] = extract
            new_token["weight"] = weight
            all_tokens.append(new_token)

    df = pd.DataFrame(all_tokens)
    df.to_csv(features_csv_filename)


def convertTextFileToGzip(file_in, file_out):
    with open(file_in, "r") as f_in:
        with gzip.open(file_out, "wt") as f_out:
            for line in f_in:
                f_out.write(line)


def test():
    import spacy
    en_nlp = spacy.load('en_core_web_lg')
    tokens = en_nlp(u'This is a test sentence.')
    print([token.pos for token in tokens])
    print([token.dep for token in tokens])


def main():
    test()
    # exp_dir = r"/Users/masterman/NLP/PhD/aac/experiments/aac_full_text_kw"
    exp_dir = r"G:\NLP\PhD\aac\experiments\aac_full_text_kw"
    features_filename = os.path.join(exp_dir, "feature_data.json")
    features_numpy_filename = os.path.join(exp_dir, "feature_data.npy.gz")
    features_csv_filename = os.path.join(exp_dir, "feature_data.csv")

    # convert_json_to_matrix(features_filename, features_numpy_filename)
    # convert_json_to_csv(features_filename, features_csv_filename)
    convertTextFileToGzip(features_filename, features_filename + ".gz")


def test_freader():
    # convertTextFileToGzip(r"/Users/masterman/NLP/PhD/aac/experiments/aac_full_text_kw_selection/test_file.txt",
    #                   r"/Users/masterman/NLP/PhD/aac/experiments/aac_full_text_kw_selection/test_file.txt.gz", )

    # reader = FeaturesReader(r"/Users/masterman/NLP/PhD/aac/experiments/aac_full_text_kw_selection/test_file.txt")
    reader = FeaturesReader(r"/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/feature_data.json.gz", 1)

    for item in reader:
        processed_text = " ".join([t["text"] for t in item["tokens"]])
        print(processed_text)
        print("\n")
        print(item)
        # to_extract = []
        # for index, t in enumerate(item["tokens"]):
        #     # if item["extract_mask"][index]:
        #     #     to_extract.append(t["text"])
        #     if item["weight_mask"][index]:
        #         to_extract.append((t["text"], item["weight_mask"][index]))
        #
        # print(to_extract)
        # print(item["best_kws"])

    # print(len(reader))
    #
    # for item in reader.getIterator(1):
    #     print(json.dumps(item, indent=3), "\n\n")

    # print(len(reader))
    # for line in reader.getMiniBatch(5):
    #     print(line)


if __name__ == '__main__':
    # main()
    test_freader()


def flattenList(items):
    """
    Concatenates all tokens from all contexts into a single list

    :param items:
    :return:
    """
    res = []
    for context in items:
        res.extend(context)

    return res


def tokenWeight(token):
    if len(token) == 3:
        return token[2]
    else:
        return token[1]


def measurePR(truth, predictions):
    if len(truth) == 0:
        return 0, 0

    tp = fp = fn = 0
    for word in predictions:
        if word in truth:
            tp += 1
        else:
            fp += 1

    for word in truth:
        if word not in predictions:
            fn += 1

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    return precision, recall, tp, (tp + fp), (tp + fn)


def getRootDir(subdir=None):
    """
    Returns the root directory for data storage (index, json files, etc.) based on where
    the script is being run

    :return:
    """
    dir_options = ["g:\\nlp\\phd",
                   "c:\\nlp\\phd",
                   "/Users/masterman/NLP/PhD",
                   # "/home/iomasterman_gmail_com",
                   "/afs/inf.ed.ac.uk/user/s11/s1135029"]
    for option in dir_options:
        if os.path.isdir(option):
            if not subdir:
                return option
            else:
                return os.path.join(option, subdir)


def statsOnPredictions(results_predictions, stopwords):
    unique_kws_extracted = []
    num_kws_extracted = []
    total_terms = []

    for precomputed_query in results_predictions:
        if "vis_text" in precomputed_query:
            original_query = precomputed_query.get("vis_text")
        elif "query_text" in precomputed_query and precomputed_query["query_text"] != "":
            original_query = precomputed_query.get("query_text")
        else:
            original_query = " ".join((t[0] + " ") * t[1] for t in precomputed_query["structured_query"])

        original_query = original_query.replace("__cit", " ")
        original_query = original_query.replace("__author", " ")
        original_query = original_query.replace("__ref", " ")
        original_query = original_query.replace("#stop#", " ")

        all_tokens = re.findall(r"\w+", original_query.lower())
        counts = Counter(all_tokens)

        total_terms.append(len(all_tokens))

        terms = [t[0].lower() for t in precomputed_query["structured_query"]]
        terms = [t for t in terms if len(t) > 1 and t not in stopwords]

        unique_kws_extracted.append(len(terms))
        num_kws_extracted.append(sum([counts[term] for term in terms]))

    unique_kws_extracted = sum(unique_kws_extracted)
    num_kws_extracted = sum(num_kws_extracted)
    total_terms = sum(total_terms)

    if total_terms > 0:
        ratio = num_kws_extracted / float(total_terms)
    else:
        ratio = 0

    print("unique_kws_extracted", unique_kws_extracted)
    print("num_kws_extracted", num_kws_extracted)
    print("total_terms", total_terms)
    print("")
    print("ratio of kws extracted", ratio)


def plotInformativeFeatures(importances, num_features, std, feature_names, indices, corpus_label, exp_dir):
    # Plot the feature importances of the forest
    plt.figure(figsize=(14, 10), dpi=300)
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.title("Feature importances for %s" % corpus_label, fontsize=20)
    plt.bar(list(range(num_features)), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(list(range(num_features)), [feature_names[index] for index in indices], rotation=90, fontsize=14)
    plt.xlim([-1, num_features])

    if False:
        plt.show()
    else:
        filename = os.path.join(exp_dir, "feature_importance_%s.png" % os.path.basename(exp_dir))
        plt.savefig(filename)


bins = [
    (0, 0.02),
    (0.02, 0.05),
    (0.05, 0.1),
    (0.1, 0.15),
    (0.15, 0.2),
    (0.2, 0.25),
    (0.25, 0.3),
    (0.3, 0.4),
    (0.4, 0.1)]


def makeBins(context):
    bin_mask = []

    for index, weight in enumerate(context["weight_mask"]):
        if weight == 0:
            bin_mask.append(0)
            continue

        for bindex in range(len(bins)):
            if weight > bins[bindex][0] and weight <= bins[bindex][1]:
                bin_mask.append(bindex + 1)

    context["bin_mask"] = bin_mask
