from __future__ import division

import json
import os
import re

from collections import Counter
from copy import deepcopy

from models.keyword_features import statsOnPredictions
from proc.stopword_proc import getStopwords
import sys


def getCorpusFromPath(exp_dir):
    if "aac" in exp_dir:
        corpus = "aac"
    elif "pmc" in exp_dir:
        corpus = "pmc"
    else:
        corpus = "aac"

    return corpus


def statsOnPredictionsFile(filename):
    with open(filename, "r") as f:
        queries = json.load(f)

    statsOnPredictions(queries, getStopwords(getCorpusFromPath(filename)))


def externalTest(exp_dir, filename):
    # import subprocess
    statsOnPredictionsFile(os.path.join(exp_dir, filename))

    command = "python3 " + os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])),
                                        "kw_evaluation_runs",
                                        "evaluate_kw_selection.py")
    print(command)
    corpus = getCorpusFromPath(exp_dir)
    command += " " + corpus + " " + os.path.basename(os.path.dirname(exp_dir)) + " " + filename
    os.system(command)


def get_sentence_tokens_from_token_list(tokens):
    cur_sentence = []
    sentences = []
    for token in tokens:
        if token["text"] == "#STOP#":
            sentences.append(cur_sentence)
            cur_sentence = []
        else:
            cur_sentence.append(token)

    return sentences


def get_sentence_text_from_tokens(tokens):
    all_text = " ".join([x["text"] for x in tokens])
    sentence = all_text.strip()
    sentence = sentence.replace(" ,", ",").replace(" .", ".")
    if sentence == "":
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


def getContextText(context):
    sentences = get_sentence_tokens_from_token_list(context["tokens"])
    sents = [get_sentence_text_from_tokens(sent) for sent in sentences]
    text = " ".join(sents)
    return text


# def getAnnotatedItemsFromFile(filename):


def generateAnnotationFile(contexts, filename, existing_kps={}):
    f = open(filename, "w")
    done_queries = {}
    for index, context in enumerate(contexts):
        query_id = context["file_guid"] + "_" + context["citation_id"]
        if query_id in done_queries:
            continue
        # text = getContextText(context)

        kws = ", ".join(existing_kps.get(query_id, []))
        text = context["vis_text"]
        text = text.replace("__cit", "[CITATION HERE]")
        text = text.replace("__trace", "[other cit]")
        text = text.replace("'", "''")

        f.write("#ID: " + query_id + "\n")
        f.write("#NUM: %d\n\n" % index)
        f.write("#WORDS: " + text + "\n\n")
        f.write("#KEYWORDS: %s \n\n\n" % kws)
        done_queries[query_id] = True


def getAnnotFilename(queries_filename):
    annot_dir = os.path.dirname(queries_filename)
    annot_filename = os.path.join(annot_dir, os.path.splitext(os.path.basename(queries_filename))[0] + "_annot.txt")
    return annot_filename


def generateAnnotationFileForQueries(queries_filename, existing_annotation_file=None):
    queries = json.load(open(queries_filename, "r"))
    annot_filename = getAnnotFilename(queries_filename)

    existing_kps = {}
    if existing_annotation_file:
        existing_kps, _ = readAnnotationFile(existing_annotation_file)

    generateAnnotationFile(queries, annot_filename, existing_kps)


def readKeyphrases(words):
    """
    DEPRECATED. Use to read "keyphrases", separated by commas

    :param words:
    :return:
    """
    kps = [kp.strip() for kp in words.split(",")]
    mod_kps = []
    for kp in kps:
        kp = re.sub(r"[.\\/()=-]+", " ", kp)
        mod_kps.append(kp)
    kps = mod_kps
    return kps


def readKeywords(words):
    """
    Reads keywords from "#KEYWORDS:", separated by any punctuation

    :param words:
    :return:
    """
    words = re.sub(r"[.\\/()=-]+", " ", words)
    kws = words.split()
    return kws


def readAnnotationFile(annot_filename):
    all_kps = {}
    all_items = {}


    with open(annot_filename, "r") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if line.startswith("#ID"):

                match = re.search(r"#ID: (.+)", line)
                if not match:
                    print("Error on line", index)
                    raise ValueError
                current_id = match.group(1)
                all_items[current_id] = {"id": current_id, "words": "", "keywords": "",
                                         "keyphrases": "", "num": 0}
            elif line.startswith("#KEYWORDS"):
                match = re.search(r"#KEYWORDS(\w?): (.+)", line)
                annotator = None
                if match:
                    # words = match.group(2).lower()
                    keywords = match.group(2)
                    kps = readKeyphrases(keywords)
                    kws = readKeywords(keywords)
                    if match.group(1):
                        annotator = int(match.group(1))
                else:
                    kps = []
                    kws = []
                    keywords = ""
                all_kps[current_id] = kps

                keywords_key = "keywords"

                if not annotator:
                    if len(all_items[current_id]["keywords"]) > 0:
                        annotator = 2
                        keywords_key = "keywords" + str(annotator)
                    else:
                        annotator = 1

                all_items[current_id][keywords_key] = kws
                all_items[current_id]["original_" + keywords_key] = keywords

            elif line.startswith("#NUM"):
                all_items[current_id]["num"] = line.replace("#NUM", "").strip()
            elif line.startswith("#WORDS"):
                all_items[current_id]["words"] = line.replace("#WORDS", "").strip()

    return all_kps, all_items


def makeEvaluationQuery(context, manual_kw, manual_kp):
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

    counts = Counter([t[0].lower() for t in context["structured_query"]])

    query = []

    for kw in manual_kw:
        # (text, count, boost, bool, field, distance)
        new_token = (kw, counts[kw], 1, None, None, None)
        # new_token = (text, counts[text], float(weight)/counts[text], None, None, None)
        query.append(new_token)

    result_context["structured_query"] = query

    result_context["structured_query_manual"] = query
    result_context["keyphrases_manual"] = manual_kp
    result_context["manual_annotation"] = True
    return result_context


def addManualKPsToQueries(queries_filename, annot_filename=None):
    queries = json.load(open(queries_filename, "r"))

    if not annot_filename:
        annot_filename = getAnnotFilename(queries_filename)
    all_kps, _ = readAnnotationFile(annot_filename)

    q_dir = os.path.dirname(queries_filename)
    q_name = os.path.splitext(os.path.basename(queries_filename))[0]
    annot_q_file = os.path.join(q_dir, q_name + "_annot.json")

    res = []
    stopwords = getStopwords(q_dir)

    for query in queries:
        query_id = query["file_guid"] + "_" + query["citation_id"]
        context_kw = []
        context_kp = []
        kws = []
        for kp in all_kps[query_id]:
            kws = re.split("\W+", kp.strip())
            kws = [kw.lower() for kw in kws if kw not in stopwords and len(kw) > 2]
            context_kw.extend(kws)
            if " " in kp:
                context_kp.append(kp)

        if len(kws) == 0:
            continue

        print(context_kw)
        new_q = makeEvaluationQuery(query, context_kw, context_kp)
        res.append(new_q)

    json.dump(res, open(annot_q_file, "w"))
    print("Queries written:", len(res))


def makeBaselineQueries(queries_filename):
    stopwords = getStopwords(getCorpusFromPath(queries_filename))
    queries = json.load(open(queries_filename, "r"))

    for precomputed_query in queries:
        original_query = precomputed_query["vis_text"]
        original_query = original_query.replace("__cit", " ")
        original_query = original_query.replace("__author", " ")
        original_query = original_query.replace("__ref", " ")

        all_tokens = re.findall(r"\w+", original_query.lower())
        terms = list(set(all_tokens))
        counts = Counter(all_tokens)
        terms_to_extract = [t for t in terms if len(t) > 1 and t not in stopwords]
        precomputed_query["structured_query"] = [(t, counts[t], 1) for t in terms_to_extract]

    json.dump(queries, open(queries_filename.replace(".json", "_baseline.json"), "w"))


def main_aac():
    # generateAnnotationFileForQueries(
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/precomputed_queries_new1k.json",
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/precomputed_queries_new1k_annot.txt"
    # )
    #
    # addManualKPsToQueries(
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/precomputed_queries_new1k.json")
    #
    # makeBaselineQueries(
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/precomputed_queries_new1k_annot.json")

    externalTest("/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/",
                 "precomputed_queries_new1k_annot.json")

    externalTest("/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/",
                 "precomputed_queries_new1k_annot_baseline.json")


def main_pmc():
    # generateAnnotationFileForQueries(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/precomputed_queries_test.json",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/precomputed_queries_test_annot.txt"
    # )

    addManualKPsToQueries(
        "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/precomputed_queries_test.json")

    makeBaselineQueries(
        "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/precomputed_queries_test_annot.json")

    externalTest("/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/",
                 "precomputed_queries_test_annot.json")

    externalTest("/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/",
                 "precomputed_queries_test_annot_baseline.json")


if __name__ == '__main__':
    # main_aac()
    main_pmc()
