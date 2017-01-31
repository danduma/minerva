# DocumentFeaturesAnnotator: annotates all features in a document to be used for
# keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json
from collections import Counter
from math import sqrt, log

from sklearn.feature_extraction.text import TfidfVectorizer
import requests

from minerva.proc.nlp_functions import (stopwords, replaceCitationTokensForParsing,
getCitationNumberFromToken, getFirstNumberFromString)
import minerva.db.corpora as cp

from elasticsearch import ConnectionError

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
    return docs_in_index

def getElasticTermScores(token_string, index_name, field_names, doc_type="doc"):
    """
        Returns doc_freq and ttf (total term frequency) from the elastic index
        for each field for each token

        :param token_string: a string containing all the tokens
        :index_name: name of the index we're querying
        :field_names: a list of fields to query for
        :returns: a dict[field][token]{
    """
    if isinstance(cp.Corpus.endpoint, basestring):
        url=cp.Corpus.endpoint+index_name+"/"+doc_type+"/_termvectors"
    else:
        url="http://%s:%d/%s/%s/_termvectors" % (cp.Corpus.endpoint["host"],
                                                 cp.Corpus.endpoint["port"],
                                                 index_name,
                                                 doc_type
                                                 )

    if isinstance(field_names,basestring):
        field_names=[field_names]

    doc={field_name:token_string[:200] for field_name in field_names}
    request={
             "term_statistics" : True,
             "field_statistics" : False,
             "positions" : False,
             "offsets" : False,
             "payloads" : False,
             "dfs": True,
             "doc" : doc
    }
    r=requests.post(url,data=json.dumps(request))
    data=json.loads(r.text)

    res={}
    if "error" in data:
        print(data)
        raise ConnectionError(data["error"]["reason"])

    if data["found"]:
        for field in data["term_vectors"]:
            res[field]={}
            for token in data["term_vectors"][field]:
                res[field][token]=data["term_vectors"][field][token]
##                del res[field][token]["term_freq"]

##    assert False
    return res

def getCitationTokenPosition(cit_num, tokens):
    """
        Returns the token number in a list of tokens for the cit_id we seek
    """
    for index, token in enumerate(tokens):
        if token["cit_num"]==cit_num:
            return index

    return None

def annotateTokenDistanceFromCitations(doc,
                                       sentence_window=5,
                                       token_window=200,
                                       only_within_paragraph=False,
                                       normalize_values=False):
    """
        Adds to each token the distance to each citation in the document, within
        a window of sentences
    """
    for cit in doc.citations:
        sent_id=getFirstNumberFromString(cit["parent_s"])
        if only_within_paragraph:
            para=doc.element_by_id[doc.element_by_id[cit["parent_s"]]["parent"]]
            min_sent=getFirstNumberFromString(para["content"][0]) # id of first sentence in paragraph
            max_sent=getFirstNumberFromString(para["content"][0]) # id of first sentence in paragraph
        else:
            min_sent=max(0,sent_id-sentence_window)
            max_sent=min(len(doc.allsentences)-1,sent_id-sentence_window)

        sents=doc.allsentences[min_sent:max_sent]
        all_tokens=[]
        for sent in sents:
            all_tokens.extend(sent["token_features"])

        cit_num=getCitationNumberFromToken(cit["id"])
        cit_pos=getCitationTokenPosition(cit_num, all_tokens)

        dict_key="dist_cit_"+str(cit_num)
        for cnt in range(cit_pos-min(cit_pos,token_window),cit_pos+min(len(all_tokens)-cit_pos,token_window)):
            val=cnt-cit_pos
            if normalize_values:
                val=val/float(token_window)
            all_tokens[cnt][dict_key]=val


class DocumentFeaturesAnnotator(object):
    """
        Pre-annotates all features in a scidoc to be used for keyword extraction
    """
    def __init__(self, index_name, doc_type="doc"):
        self.index_name=index_name
        self.doc_type=doc_type

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
            parse = en_nlp(replaceCitationTokensForParsing(sent["text"]))
            all_parses[sent["id"]]=parse

        tokens=[]
        for parse in all_parses.values():
            for token in parse:
                tokens.append(token.lower_)

##        tf_scores=get_tf_scores(tokens)
        df_scores=getElasticTermScores(doctext,self.index_name,"text")
        if len(df_scores) > 0:
            df_scores=df_scores["text"]
        numDocs=getElasticTotalDocs(self.index_name, self.doc_type)

        for sent in doc.allsentences:
            token_features=[]
            text=token.lower_
            parse=all_parses[sent["id"]]
            for token in parse:
                tf=df_scores[text]["term_freq"]
                df=df_scores[text]["doc_freq"]
                tf_idf=sqrt(tf) * (1 + log(numDocs/float(df+1)))

                features={"text":text,
                          "pos":token.pos,
                          "dep":token.dep,
                          "ent_type":token.ent_iob,
                          "lemma":token.lemma_,
                          "is_punct":token.is_punct,
                          "is_lower":token.is_lower,
                          "like_num":token.like_num,
                          "is_stop":token.is_stop,
                          "tf":tf,
                          "df":df,
                          "tf_idf":tf_idf,
                          "token_type":"t",# "t" for normal token, "c" for citation, "p" for punctuation
                          # dis_cit_XX : for each citation within a window of sentences,
                          }

                cit_id=getCitationNumberFromToken(token.lower_)
                if cit_id:
                    features["cit_num"]=cit_id
                    features["token_type"]="c"
                elif token.is_punct:
                    features["token_type"]="p"

                token_features.append(features)
            sent["token_features"]=token_features

        annotateTokenDistanceFromCitations(doc, token_window=300, only_within_paragraph=True, normalize_values=True)
        return doc

def main():
    pass

if __name__ == '__main__':
    main()
