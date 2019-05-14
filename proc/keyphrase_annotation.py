import os
import nltk

from jgtextrank import keywords_extraction
# load custom stop list
# The SMART stop-word list built by Chris Buckley and Gerard Salton,
#   which can be obtained from http://www.lextek.com/manuals/onix/stopwords2.html
import db.corpora as cp

for stoplist_path in [os.path.join("..", "notebooks", 'smart-stop-list.txt'),
                      os.path.join("notebooks", 'smart-stop-list.txt')]:
    if os.path.exists(stoplist_path):
        with open(stoplist_path) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        break

stop_list = lines

from string import punctuation

stop_list.extend(punctuation)
stop_list.extend(["__author", "__cit", "table", "figure"])
stop_list = set(stop_list)


# 'max' : maximum value of vertices weights
# 'avg' : avarage vertices weight
# 'sum' : sum of vertices weights
# 'norm_max' : MWT unit size normalisation of 'max' weight
# 'norm_avg' : MWT unit size normalisation of 'avg' weight
# 'norm_sum' : MWT unit size normalisation of 'sum' weight
# 'log_norm_max' : logarithm based normalisation of 'max' weight
# 'log_norm_avg' : logarithm based normalisation of 'avg' weight
# 'log_norm_sum' : logarithm based normalisation of 'sum' weight
# 'gaussian_norm_max' : gaussian normalisation of 'max' weight
# 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
# 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight

# class KeyPhraseAnnotator(object):
#     def __init__(self):
#         pass

def tokenizeForKP(text):
    tokens = nltk.word_tokenize(text.lower())
    return tokens


def getDictsFromLists(kp1, kp2):
    d1 = {k[0]: k[1] for k in kp1}
    d2 = {k[0]: k[1] for k in kp2}
    all_keys = []
    all_keys.extend([k for k in d1.keys()])
    all_keys.extend([k for k in d2.keys()])
    all_keys = list(set(all_keys))
    return d1, d2, all_keys


def joinKPs(kp1, kp2):
    d1, d2, all_keys = getDictsFromLists(kp1, kp2)
    res = []

    for key in all_keys:
        value = 0
        if key in d1:
            value += d1[key]
        if key in d2:
            value += d2[key]
        res.append((key, value))

    return res


def sortKPs(kp_list):
    kp = sorted(kp_list, key=lambda x: x[1], reverse=True)
    return kp


def getNgramFrequency(text, ngram_length, min_freq=2, tokens=None, remove_stopwords=True):
    if not tokens:
        tokens = tokenizeForKP(text)

    ng = nltk.ngrams(tokens, ngram_length)

    # compute frequency distribution for all the ngrams in the text
    fdist = nltk.FreqDist(ng)
    kp = [i for i in fdist.items() if i[1] >= min_freq]
    kp2 = []

    if remove_stopwords:
        for i in kp:
            ignore = False
            for word in i[0]:
                if word in stop_list:
                    ignore = True
                    break
            if not ignore:
                kp2.append(i)
        kp = kp2

    kp = sortKPs(kp)
    return kp


def getDocFreqKPs(doc):
    """
    Returns KPs by frequency, counting the title and the abstract as special
    cases

    :param doc:
    :return:
    """
    kp_title = getNgramFrequency(doc.metadata["title"], 3, 1)
    kp_title.extend(getNgramFrequency(doc.metadata["title"], 2, 1))

    abstract = doc.getAbstract()
    kp_abstract = getNgramFrequency(abstract, 3, 1)
    kp_abstract.extend(getNgramFrequency(abstract, 2, 1))

    res = joinKPs(kp_title, kp_abstract)

    doc_text = doc.formatTextForExtraction(doc.getFullDocumentText(exclude_abstract=True))
    kp_body = getNgramFrequency(doc_text, 3, 2)
    kp_body.extend(getNgramFrequency(doc_text, 2, 2))

    res = joinKPs(res, kp_body)

    res = sortKPs(res)
    return res


def getTextRankKPs(text, window_size=2, mu=3, max_kp=100):
    method = 'log_norm_avg'
    method = 'max'
    # method = 'avg'
    results_wc1, top_vertices = keywords_extraction(text,
                                                    weight_comb=method,
                                                    top_p=1,
                                                    mu=mu,
                                                    lemma=True,
                                                    window=window_size,
                                                    stop_words=stop_list,
                                                    solver="pagerank_numpy",
                                                    directed=True)
    res = results_wc1[:max_kp]
    res = [(tuple(k[0].split()), k[1]) for k in res]
    return res


def getKPIntersection(kp1, kp2):
    d1, d2, all_keys = getDictsFromLists(kp1, kp2)
    res = []
    for key in all_keys:
        if key in d1 and key in d2:
            res.append(key)

    res = sortKPs(res)
    return res


def print_keyphrases(results):
    return [k[0] for k in results]


def test_different_tr_weight_comb(text, window_size=2, mu=3, max_kp=100):
    for method in ['max', 'avg',
                   # 'sum',
                   'norm_max', 'norm_avg',
                   # 'norm_sum',
                   'log_norm_max',
                   'log_norm_avg',
                   # 'log_norm_sum',
                   # 'gaussian_norm_max', 'gaussian_norm_avg', 'gaussian_norm_sum'
                   ]:
        results_wc1, top_vertices = keywords_extraction(text,
                                                        weight_comb=method,
                                                        top_p=1,
                                                        mu=mu,
                                                        lemma=True,
                                                        window=window_size,
                                                        stop_words=stop_list,
                                                        solver="pagerank_numpy",
                                                        directed=True)

        print(method, " : ", print_keyphrases(results_wc1[:max_kp]), "\n")


def testFrequencyKeyphrases(text):
    tokens = tokenizeForKP(text)

    kp2 = getNgramFrequency(text, 2, tokens=tokens, remove_stopwords=True)
    kp3 = getNgramFrequency(text, 3, tokens=tokens, remove_stopwords=True)

    for k, v in kp3[:50]:
        print(k, v)
    for k, v in kp2[:50]:
        print(k, v)


def getMatchingKPsInContext(text, all_kps):
    """
    Returns the KPs matching in the text provided

    :param text:
    :param all_kps:
    :return:
    """
    tokens = tokenizeForKP(text)
    ngrams = getNgramFrequency(text, 3, 1, tokens=tokens)
    ngrams.extend(getNgramFrequency(text, 2, 1, tokens=tokens))
    ngram_set = set([n[0] for n in ngrams])
    all_kps = [tuple(kp) for kp in all_kps]
    intersection = set(ngram_set).intersection(all_kps)
    return list(intersection)


def getFreqTextRankKeyPhrases(doc):
    """
    Filters TextRank results by frequency in the document, returning a de-noised version of the KPs.

    :param doc: scidoc
    :return: list of KP tuples without scores
    """
    sect = getDocFreqKPs(doc)
    print("From document frequency:")
    print([(" ".join(kp[0]), kp[1]) for kp in sect])
    text = doc.formatTextForExtraction(doc.getFullDocumentText())
    try:
        tr = getTextRankKPs(text, window_size=2, mu=3, max_kp=300)
    except KeyError as e:
        print("TextRank crashed. Using keyphrases appearing >2 times")
        tr = [k for k in sect if k[1] > 2]
    print("")
    print("From TextRank:")
    print([(" ".join(kp[0]), kp[1]) for kp in tr])
    kps = getKPIntersection(tr, sect)
    print("\n\n")
    print(kps)
    return kps


def annotateDocWithKPs(guid):
    corpus = cp.Corpus
    doc = corpus.loadSciDoc(guid)
    KPs = getFreqTextRankKeyPhrases(doc)
    doc.data["annotated_keyphrases"] = KPs
    corpus.saveSciDoc(doc)


if __name__ == '__main__':
    main()
