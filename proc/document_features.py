# DocumentFeaturesAnnotator: annotates all features in a document to be used for
# keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from sklearn.feature_extraction.text import TfidfVectorizer
from minerva.proc.nlp_functions import stopwords
import minerva.db.corpora as cp
import requests
from collections import Counter
from math import log

import spacy
en_nlp = spacy.load('en')

def get_tf_scores(tokens):
    """
        Returns the TF component for each term
    """
    scores=Counter(tokens)
##    for token in tokens:
##        scores[token]=scores.get(token,0)+1
    return scores


def getElasticTotalDocs(index_name, doc_type):
    """
        Returns total number of documents in an index
    """
    url="http://%s:%d/%s/%s/" % (cp.Corpus.endpoint["host"],
                                             cp.Corpus.endpoint["port"],
                                             index_name,
                                             doc_type
                                             )
    r=requests.get(url+"_count",{ "query": { "match_all": {} } })
    data=json.loads(r.text)
    docs_in_index=int(data["count"])

def getElasticTermScores(token_string, index_name, field_names, doc_type="doc"):
    """
        Returns doc_freq and ttf (total term frequency) from the elastic index
        for each field for each token

        :param token_string: a string containing all the tokens
        :index_name: name of the index we're querying
        :field_names: a list of fields to query for
        :returns: a dict[field][token]{
    """
    url="http://%s:%d/%s/%s/_termvectors" % (cp.Corpus.endpoint["host"],
                                             cp.Corpus.endpoint["port"],
                                             index_name,
                                             doc_type
                                             )

    doc={field_name:token_string for field_name in field_names}
    request={
             "term_statistics" : True,
             "field_statistics" : False,
             "positions" : False,
             "offsets" : False,
             "payloads" : False,
             "dfs": True,
             "doc" : doc
    }
    r=requests.get(index_url,request)
    data=json.loads(r.text)

    res={}
    if data["found"]:
        for field in data["term_vectors"]:
            res[field]={}
            for token in data["term_vectors"][field]:
                res[field][token]=data["term_vectors"][field][token]
##                del res[field][token]["term_freq"]
    return res

class DocumentFeaturesAnnotator(object, index_name):
    """
        Pre-annotates all features in a scidoc to be used for keyword extraction
    """
    def __init__(self):
        self.index_name=index_name

    def annotate_section(self, section):
        """
        """

    def annotate_sentence(self, s):
        """
        """

    def select_sentences(self, doc, window_size=3):
        """
            Selects which sentences to annotate
        """

        to_annotate=[0 for s in range(len(doc.allsentences))]
        for index, s in enumerate(doc.allsentences):
            if len(s.get("citations",[])) > 0:
                for cnt in range (max(index-2,0),min(index+2,len(to_annotate))+1):
                    to_annotate[cnt] = True

    def annotate(self, doc):
        """
            Adds all features to a scidoc
        """
        all_parses={}
        doctext=doc.getFullDocumentText(True, False)
        vector=TfidfVectorizer(decode_error="ignore", stop_words=stopwords)
        matrix=vector.fit_transform([doctext])
##        tf = vector.tf_
##        tf_scores=dict(zip(vector.get_feature_names(), tf))
        for sent in doc.allsentences:
            parse = en_nlp(sent["text"])
            all_parses[sent["id"]]=parse

        tokens=[]
        for parse in all_parses.values():
            for token in parse:
                tokens.append(token.lower_)

        tf_scores=get_tf_scores(tokens)
        df_scores=getElasticTermScores(doctext,self.index_name,"text")
        if len(df_scores) > 0:
            df_scores=df_scores["text"]

        for sent in doc.allsentences:
            token_features=[]
            parse=all_parses[sent["id"]]
            for token in parse:
                text=token.lower_
                tf=tf_scores[text]
                df=df_scores[text]
                tf_idf=sqrt(tf) * (1 + log(numDocs/(df+1)))

                features={"text":token.lower_,
                          "pos":token.pos,
                          "dep":token.dep,
                          "ent_type":token.ent_iob,
                          "lemma":token.lemma_,
                          "is_punct":token.is_punct,
                          "is_lower":token.is_punct,
                          "like_num":token.like_num,
                          "is_stop":token.like_num,
                          "tf":tf,
                          "df":df,
                          "tf_idf":tf_idf,
##                          "csc_type":sent.get("csc_type",""),
##                          "az":sent.get("az",""),
                          }

def main():
    pass

if __name__ == '__main__':
    main()
