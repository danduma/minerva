import os

import time
import multiprocessing

from jgtextrank import keywords_extraction, keywords_extraction_from_corpus_directory
# load custom stop list
# The SMART stop-word list built by Chris Buckley and Gerard Salton,
#   which can be obtained from http://www.lextek.com/manuals/onix/stopwords2.html
from db.ez_connect import ez_connect

with open(os.path.join("..","notebooks",'smart-stop-list.txt')) as f:
    lines=f.readlines()
    lines=[line.strip() for line in lines]

stop_list=lines

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

def print_keyphrases(results):
    return [k[0] for k in results]

def print_out_different_weight_comb(text, window_size=2, mu=3, max_kp=10):
    for method in ['max', 'avg', 'sum', 'norm_max', 'norm_avg', 'norm_sum', 'log_norm_max',
                  'log_norm_avg','log_norm_sum','gaussian_norm_max', 'gaussian_norm_avg',
                  'gaussian_norm_sum']:
        results_wc1, top_vertices = keywords_extraction(text,
                                                        weight_comb=method,
                                                        top_p = 1,
                                                        mu=mu,
                                                        lemma=True,
                                                        window=window_size, stop_words=stop_list)

        print(method," : ", print_keyphrases(results_wc1[:max_kp]), "\n")

def main():
    corpus=ez_connect("AAC")
    guids=corpus.listPapers(max_results=10)
    for guid in guids:
        doc=corpus.loadSciDoc(guid)
        text = doc.formatTextForExtraction(doc.getFullDocumentText())
        print(doc.metadata["title"])
        print_out_different_weight_comb(text)

if __name__ == '__main__':
    main()