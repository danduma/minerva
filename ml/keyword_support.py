import gzip
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from proc.results_logging import ProgressIndicator
from db.result_store import OfflineResultReader
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
    "dist_cit": 1000
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
    "dist_cit": 1000
}

token_type_to_int = {"t": 0, "c": 1, "p": 2}
az_to_int = {"": 0}
coresc_to_int = {"": 0}

for index, az in enumerate(AZ_ZONES_LIST):
    az_to_int[az] = index

for index, csc in enumerate(CORESC_LIST):
    coresc_to_int[csc] = index


def saveFeatureData(precomputed_contexts, path):
    """
        Calls prepareFeatureData() and dumps prepared data to json file
    """
    feature_data = prepareFeatureData(precomputed_contexts)

    if path.endswith(".gz"):
        f = gzip.open(path, "wt")
    else:
        f = open(path, "w")

    for line in feature_data:
        for sentence in feature_data:
            for packed_token in sentence:
                assert packed_token[0]["pos"] != 0
        f.write(json.dumps(line) + "\n")

    f.close()
    return feature_data


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


def saveKeywordSelectionScores(reader, exp_dir):
    """
        Saves a CSV to measure the performance of keyword selection
    """

    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        return {
            "precision_score": kw_data["precision_score"],
            "mrr_score": kw_data["mrr_score"],
            "rank": kw_data["rank"],
            "ndcg_score": kw_data["ndcg_score"],

            "precision_score_kw": kw_data["kw_selection_scores"]["precision_score"],
            "mrr_score_kw": kw_data["kw_selection_scores"]["mrr_score"],
            "rank_kw": kw_data["kw_selection_scores"]["rank"],
            "ndcg_score_kw": kw_data["kw_selection_scores"]["ndcg_score"],

            "precision_score_kw_weight": kw_data["kw_selection_weight_scores"]["precision_score"],
            "mrr_score_kw_weight": kw_data["kw_selection_weight_scores"]["mrr_score"],
            "rank_kw_weight": kw_data["kw_selection_weight_scores"]["rank"],
            "ndcg_score_kw_weight": kw_data["kw_selection_weight_scores"]["ndcg_score"],
        }

    lines = []
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data = pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir, "kw_selection_scores.csv"))


def saveAllKeywordSelectionTrace(reader, exp_dir):
    """
        Saves a CSV to trace everything that happened, inspect the dataset
    """

    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        context = kw_data["context"]
        if isinstance(context, list):
            context = u" ".join([s["text"] for s in context])

        res = {
            "cit_ids": kw_data["cit_ids"],
            "cit_multi": kw_data["cit_multi"],
            "context": context,
            "file_guid": kw_data.get("file_guid", ""),
            "match_guids": kw_data["match_guids"],
            "best_kws": [kw[0] for kw in kw_data["best_kws"]],

            "precision_score": kw_data["precision_score"],
            "mrr_score": kw_data["mrr_score"],
            "rank": kw_data["rank"],
            "ndcg_score": kw_data["ndcg_score"],

            "precision_score_kw": kw_data["kw_selection_scores"]["precision_score"],
            "mrr_score_kw": kw_data["kw_selection_scores"]["mrr_score"],
            "rank_kw": kw_data["kw_selection_scores"]["rank"],
            "ndcg_score_kw": kw_data["kw_selection_scores"]["ndcg_score"],

            "precision_score_kw_weight": kw_data["kw_selection_weight_scores"]["precision_score"],
            "mrr_score_kw_weight": kw_data["kw_selection_weight_scores"]["mrr_score"],
            "rank_kw_weight": kw_data["kw_selection_weight_scores"]["rank"],
            "ndcg_score_kw_weight": kw_data["kw_selection_weight_scores"]["ndcg_score"],
        }
        if "keyword_selection_entry" in kw_data:
            res["keyword_selection_entry"] = kw_data["keyword_selection_entry"]
        return res

    lines = []
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data = pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir, "kw_selection_trace.csv"), encoding="utf-8")


def saveOfflineKWSelectionTraceToCSV(reader_name, cache_dir, output_dir):
    """
    Assumes all data has been cached locally, processes and generates CSV from it

    :param cache_dir: where the cache was saved
    :param output_dir: directory where the CSV output will be saved
    :return:
    """

    reader = OfflineResultReader(reader_name, cache_dir)
    ##            print("\n From OfflineResultReader:",listAllKeywordsToExtractFromReader(self.reader))
    ##            saveKeywordSelectionScores(self.reader, self.exp["exp_dir"])
    saveAllKeywordSelectionTrace(reader, output_dir)


def getTokenListFromContexts(contexts):
    """
    Returns a single list of tokens form all of the contexts in feature_data

    :param contexts: list of contexts, each of which is a list of tokens, each of which a dict of features
    :return: list
    """
    all_tokens = []
    if len(contexts) > 0 and contexts[0][0][0]["text"] != "#SENT_END#":
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
            if isinstance(lemma, str):
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
            matrix[cnt, i, 12] = token.get("dist_cit", 1000)
            matrix[cnt, i, 13] = token_type_to_int[token.get("token_type", "t")]

            matrix[cnt, i, 14] = int(extract)
            matrix[cnt, i, 15] = weight
    return matrix


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


def buildFeatureSetForContext(all_token_features, all_keywords):
    """
        Returns a list of (token_features_dict,{True,False}) tuples. For each
        token, based on its features, the token is annotated as to-be-extracted or not
    """
    res = []
    for token in all_token_features:
        extract = (token["text"] in list(all_keywords.keys()))
        weight = token.get("weight", 0.0)
        try:
            del token["weight"]
            del token["extract"]
        except:
            pass
        res.append((token, extract, weight))
    return res


def prepareFeatureData(precomputed_contexts):
    """
        Extracts just the features, prepares the data in a format ready for training
        classifiers, just one very long list of (token_features_dict, {True,False})
    """

    all_contexts_tokens = []

    progress = ProgressIndicator(True, len(precomputed_contexts), True)
    for context in precomputed_contexts:
        context_tokens = []
        all_keywords = {t[0]: t[1] for t in context["best_kws"]}
        for sentence in context["context"]:
            for feature in SENTENCE_FEATURES_TO_COPY:
                for token_feature in sentence["token_features"]:
                    token_feature[feature] = sentence.get(feature, "")
            ##            for token_feature in sentence["token_features"]:
            ##                for key in token_feature:
            ##                    if key.startswith("dist_cit_"):
            sent_tokens = deepcopy(sentence["token_features"])
            sent_tokens.append(SENTENCE_END_TOKEN)
            context_tokens.extend(sent_tokens)
        all_contexts_tokens.append(buildFeatureSetForContext(context_tokens,
                                                             all_keywords))
        progress.showProgressReport("Preparing feature data")
    return all_contexts_tokens


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


def save_json_as_gzip(features_filename, features_filename_gz):
    feature_data = loadFeatureData(features_filename)
    with gzip.open(features_filename_gz, "wt") as f:
        for line in feature_data:
            f.write(json.dumps(line) + "\n")


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
    save_json_as_gzip(features_filename, features_filename + ".gz")


if __name__ == '__main__':
    main()
