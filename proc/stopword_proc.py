from collections import Counter
from string import punctuation
from tqdm import tqdm
from retrieval.elastic_functions import getElasticTermScores

import json
import os

from models.keyword_features import FeaturesReader, getRootDir

c3_stopwords = ["~", "the", "and", "or", "not", "of", "to", "from", "by", "with", "a", "an"]

c6_stopwords_aac_30 = ['in', 'are', 'to', 'and', 'them', 'like', 'or', 'only', 'based', 'of', 'the', 'for', 'also', 'uses',
                    'on', 'within', 'will', 'be', 'more', 'than', 'from', 'us', 'with', 'can', 'used', 'is', 'found',
                    'after',
                    'we', 'it', 'need', 'as', 'may', 'by', 'if', 'has', 'no', 'any', 'would', 'this', 'some', 'all',
                    'here', 'use',
                    'very', 'been', 'shown', 'given', 'there', 'but', 'have', 'thus', 'then', 'show', 'so', 'these',
                    'before', 'into',
                    'al', 'made', 'was', 'an', 'not', 'makes', 'at', 'each', 'using', 'were', 'well', 'could', 'less',
                    'see', 'focus',
                    'about', 'out', 'take', 'most', 'often', 'over', 'since', 'does', 'where', 'both', 'they', 'rather',
                    'should', 'while', 'up', 'how', 'above', 'when', 'among', 'must', 'even', 'allows', 'still', 'do',
                    'might', 'find',
                    'called', 'under',
                    'make', 'those', 'etc', 'much', 'shows', 'done', 'being', 'below', 'either', 'note', 'during',
                    'known', 'now',
                    'due', 'i.e', 'just']

c6_stopwords_aac_40 = ['the', 'of', 'and', 'to', 'in', 'for', 'is', 'we', 'as', 'on', 'that', 'are', 'with', 'by', 'this',
                    'from', 'based', 'which', 'be', 'our', 'used', 'using', 'or', 'have', 'been', 'has', 'an', 'can',
                    'such', 'it', 'not', 'these', 'also', 'other', 'two', 'use', 'each', 'more', 'between', 'one',
                    'different', 'all', 'was', 'their', 'both', 'but', 'most', 'into', 'only', 'they', 'at', 'however',
                    'there', 'some', 'where', 'first', 'were', 'many', 'than', 'its', 'same', 'well', 'given', 'set',
                    'then', 'when', 'may', 'new', 'them', 'if', 'will', 'do', 'three', 'so', 'about', 'possible', 'no',
                    'how', 'any', 'following', 'would']

c6_stopwords_aac = c6_stopwords_aac_40
c6_stopwords_aac.extend(["~", "__author", "__ref"])
c6_stopwords_aac.extend(punctuation)

c6_stopwords_aac = set(c6_stopwords_aac)

c6_stopwords_pmc_30 = ['have', 'been', 'to', 'these', 'up', 'of', 'the', 'within', 'an', 'and', 'this', 'may', 'be',
                       'in',
                       'is', 'might', 'lead', 'by', 'we', 'will', 'some', 'more', 'about', 'most', 'based', 'on',
                       'known',
                       'as', 'that', 'no', 'under', 'then', 'are', 'from', 'if', 'found', 'there', 'can', 'taken',
                       'see',
                       'for', 'very', 'affect', 'were', 'seen', 'after', 'do', 'not', 'only', 'also', 'play', 'it',
                       'with',
                       'but', 'or', 'among', 'all', 'while', 'has', 'above', 'show', 'should', 'than', 'did', 'during',
                       'was', 'where', 'occur', 'both', 'co', 'per', 'using', 'reduce', 'due', 'added', 'those', 'used',
                       'at', 'into', 'shown', 'even', 'well', 'shows', 'so', 'they', 'often', 'out', 'assess', 'over',
                       'any', 'via', 'much', 'being', 'how', 'each', 'showed', 'ii', 'when', 'tested', 'lt', 'had',
                       'could', 'thus', 'like', 'form', 'needed', 'et', 'still', 'here', 'given', 'since', 'al', 'them',
                       'cause', 'least', 'would', 'rather', 'does', 'either', 'until', 'caused', 'before', 'highly',
                       'below', 'made', 'mm']

c6_stopwords_pmc_40 = ['are', 'for', 'but', 'to', 'the', 'in', 'among', 'should', 'not', 'only', 'into', 'and', 'also',
                       'of', 'as', 'all', 'has', 'by', 'where', 'is', 'or', 'have', 'been', 'using', 'with', 'these',
                       'from', 'since', 'due', 'show', 'that', 'were', 'both', 'was', 'shown', 'found', 'over', 'based',
                       'can', 'very', 'showed', 'on', 'each', 'we', 'well', 'known', 'be', 'at', 'either', 'most',
                       'this', 'here', 'those', 'under', 'per', 'some', 'while', 'used', 'any', 'more', 'an', 'even',
                       'highly', 'during', 'within', 'form', 'may', 'them', 'there', 'being', 'it', 'up', 'could', 'no',
                       'when', 'then', 'thus', 'than', 'given', 'they', 'will', 'would', 'after', 'still', 'had', 'did',
                       'out', 'do', 'before', 'least', 'does', 'if', 'might', 'above', 'about', 'made', 'lt', 'so',
                       'et', 'al']

c6_stopwords_pmc = c6_stopwords_pmc_40

c6_stopwords_pmc.extend(["~", "__author", "__ref"])
c6_stopwords_pmc.extend(punctuation)
c6_stopwords_pmc = set(c6_stopwords_pmc)


def getStopwords(index_name):
    if "aac" in index_name:
        stopwords = c6_stopwords_aac
    elif "pmc" in index_name or "coresc" in index_name:
        stopwords = c6_stopwords_pmc
    else:
        raise ValueError("Can't detect corpus")
    return stopwords


def getStopWordsFromTokens(tokens, term_scores, min_docs_to_match, max_docs_to_match, stats, min_term_len=0,
                           max_docfreq=None):
    """
    filter terms that don't appear in a minimum of documents across the corpus
    """
    removed = {}
    res = []

    pos_dict = stats["pos_dict"]

    local_stopwords = ["__author", "__ref"]
    local_stopwords.extend(punctuation)

    for token in tokens:
        to_remove = False

        term = token["text"].lower()

        if pos_dict[term] in ["NOUN", "ADJ"]:
            continue

        if token["like_num"]:
            continue

        if term in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero"]:
            continue

        freq = term_scores.get(term, {}).get("doc_freq", 0)
        ttf = term_scores.get(term, {}).get("ttf", 0)

        if ttf == 0 or freq == 0:
            continue

        if len(term) == 1 or len(term) > 6:
            continue

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
    return removed


def getTermScoresFromElastic(contexts,
                             endpoint,
                             index_name,
                             field_name):
    final_term_scores = {}
    max_doc_freq = 0

    for context in tqdm(contexts, desc="Loading term vectors"):
        all_context_terms = [t["text"].lower() for t in context["tokens"]]
        all_context_terms = [t for t in all_context_terms if t not in final_term_scores]

        token_string = " ".join([t for t in all_context_terms if t not in final_term_scores])
        token_string = token_string.strip()
        if not token_string:
            continue

        term_scores = getElasticTermScores(endpoint,
                                           token_string,
                                           index_name,
                                           field_name
                                           )

        term_scores = term_scores[field_name]["terms"]
        for term in term_scores:
            if term not in final_term_scores:
                final_term_scores[term] = term_scores[term]
                if term_scores[term].get("doc_freq", 0) > max_doc_freq:
                    max_doc_freq = term_scores[term]["doc_freq"]

    return final_term_scores, max_doc_freq


def getStopwordsFromContexts(contexts,
                             endpoint,
                             index_name,
                             field_name,
                             term_scores_filename,
                             stopwords_filename,
                             percent
                             ):
    max_doc_freq = 0

    if os.path.exists(term_scores_filename):
        term_scores = json.load(open(term_scores_filename, "r"))
        for term in term_scores:
            if term_scores[term].get("doc_freq", 0) > max_doc_freq:
                max_doc_freq = term_scores[term]["doc_freq"]
    else:
        term_scores, max_doc_freq = getTermScoresFromElastic(contexts, endpoint, index_name, field_name)
        json.dump(term_scores, open(term_scores_filename, "w"))

    contexts.seek(0, 0)
    pos_dict = {}
    like_num = {}
    for context in contexts:
        for token in context["tokens"]:
            term = token["text"].lower()
            if term not in pos_dict:
                pos_dict[term] = []

            pos_dict[term].append(token["pos_"])

    for term in pos_dict:
        counts = Counter(pos_dict[term])
        top = counts.most_common(1)
        pos_dict[term] = top[0][0]

    stats = {"pos_dict": pos_dict}

    contexts.seek(0, 0)

    max_docs_to_match = max_doc_freq * percent

    all_stopwords = {}

    for context in contexts:
        stopwords = getStopWordsFromTokens(context["tokens"],
                                           term_scores,
                                           min_docs_to_match=0,
                                           max_docs_to_match=max_docs_to_match,
                                           stats=stats,
                                           min_term_len=0)
        for stopword in stopwords:
            all_stopwords[stopword] = all_stopwords.get(stopword, 0) + stopwords[stopword]

    with open(stopwords_filename, "w") as f:
        sw_sorted = sorted(all_stopwords.items(), key=lambda x: x[1], reverse=True)
        for index, stopword in enumerate(sw_sorted):
            if stopword[1] < 1:
                all_stopwords = {sw[0]: sw[1] for sw in sw_sorted[:index]}
                break
            tabs = "\t"
            if len(stopword[0]) < 4:
                tabs *= 2
            f.write("%s%s%d\t%d\n" % (stopword[0], tabs, stopword[1], term_scores.get(stopword[0], {}).get("ttf", 0)))

    print(list(all_stopwords.keys()))
    return all_stopwords


def main():
    # do_corpus = "aac"
    # do_corpus = "pmc"
    do_corpus =  None

    percent = 0.4

    if do_corpus == "aac":
        exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
        features = FeaturesReader(os.path.join(exp_dir, "feature_data_ft_mms_min1.json.gz"))

        getStopwordsFromContexts(features,
                                 "http://129.215.197.75:9200/",
                                 "idx_az_ilc_az_annotated_aac_2010_1_paragraph",
                                 "_all_text",
                                 os.path.join(exp_dir, "term_scores.json"),
                                 os.path.join(exp_dir, "stopwords_aac.txt"),
                                 percent=percent
                                 )
    elif do_corpus == "pmc":
        exp_dir = os.path.join(getRootDir("pmc_coresc"), "experiments", "pmc_generate_kw_trace")
        # features = FeaturesReader(os.path.join(exp_dir, "feature_data_ft_mms_min1.json.gz"))
        features = FeaturesReader(os.path.join(exp_dir, "feature_data_at_w_min1.json.gz"))
        getStopwordsFromContexts(features,
                                 "http://129.215.197.75:9200/",
                                 "idx_az_ilc_az_annotated_pmc_2013_1_paragraph",
                                 "_all_text",
                                 os.path.join(exp_dir, "term_scores.json"),
                                 os.path.join(exp_dir, "stopwords_pmc.txt"),
                                 percent=percent
                                 )
    pass

    print("AAC", len(c6_stopwords_aac_40))
    print("PMC", len(c6_stopwords_pmc_40))


if __name__ == '__main__':
    main()
