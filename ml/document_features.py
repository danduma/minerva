# DocumentFeaturesAnnotator: annotates all features in a document to be used for
# keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
from math import sqrt, log

import db.corpora as cp

import spacy
from proc.nlp_functions import (stopwords,
                                getCitationNumberFromTokenText, CIT_MARKER, AUTHOR_MARKER, leaveOneCitationToken)
from retrieval.elastic_functions import getElasticTotalDocs, getElasticTermScores
from six.moves import range
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.symbols import ORTH, LEMMA, POS


def customize_spacy(nlp):
    """
        Adds the special cases for the tokenizer to keep __cit tokens together and recognize them
        as citations
    """
    special_case = [{ORTH: CIT_MARKER, LEMMA: CIT_MARKER, POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(CIT_MARKER, special_case)

    special_case = [{ORTH: AUTHOR_MARKER, LEMMA: AUTHOR_MARKER, POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(AUTHOR_MARKER, special_case)

    for i in range(300):
        text = u"__cit" + str(i)
        special_case = [{ORTH: text, LEMMA: u'__cit', POS: u'NOUN'}]
        nlp.tokenizer.add_special_case(text, special_case)

    return nlp


# en_nlp = spacy.load('en')

en_nlp = customize_spacy(spacy.load('en_core_web_lg'))


def selectContentWords(doc):
    token_res = []

    all_kp = []
    for chunk in doc.noun_chunks:
        kp = []
        for token in chunk:
            if token.pos_ not in ["PRON", "CCONJ", "PUNCT", "PROPN", "DET", "X"]:
                if token.text not in ["__cit", "__author"]:
                    kp.append(token.text.lower())

        kp = set(kp)
        all_kp.extend(kp)
    #         if len(kp) > 0:
    #             print(kp)

    # to_print = []
    for token in doc:
        #         to_print.append(token.text+"/"+token.pos_)
        if token.pos_ in ["VERB"]:
            if token.text not in all_kp:
                all_kp.append(token.text)

    #     print(" ".join(to_print))
    all_kp = list(set(all_kp))
    return all_kp


def get_tf_scores(tokens):
    """
        Returns the TF component for each term
    """
    scores = Counter(tokens)
    ##    for token in tokens:
    ##        scores[token]=scores.get(token,0)+1
    return scores


def getCitationTokenPosition(cit_num, tokens):
    """
        Returns the token number in a list of tokens for the cit_id we seek
    """
    for index, token in enumerate(tokens):
        if token.get("cit_num", None) == cit_num:
            return index

    ##    print("Couldn't find citation %d" % cit_num)
    return None


# def annotateTokenDistanceFromCitations(doc,
#                                        sentence_window=5,
#                                        token_window=200,
#                                        only_within_paragraph=False,
#                                        normalize_values=False):
#     """
#         Adds to each token the distance to each citation in the document, within
#         a window of sentences
#     """
#     for cit in doc.citations:
#         sent_id = getFirstNumberFromString(cit["parent_s"])
#         if only_within_paragraph:
#             para = doc.element_by_id[doc.element_by_id[cit["parent_s"]]["parent"]]
#             min_sent = getFirstNumberFromString(para["content"][0])  # id of first sentence in paragraph
#             max_sent = getFirstNumberFromString(para["content"][-1])  # id of last sentence in paragraph
#         else:
#             min_sent = max(0, sent_id - sentence_window)
#             max_sent = min(len(doc.allsentences) - 1, sent_id + sentence_window)
#
#         sents = doc.allsentences[min_sent:max_sent]
#         all_tokens = []
#         for sent in sents:
#             all_tokens.extend(sent["token_features"])
#
#         cit_num = getFirstNumberFromString(cit["id"])
#         cit_pos = getCitationTokenPosition(cit_num, all_tokens)
#
#         if not cit_pos:
#             ##            print("Error: couldn't find citation ",cit["id"]) # not really an error, it's just that the citation is outside of the window
#             continue
#
#         dict_key = "dist_cit_" + str(cit_num)
#         for cnt in range(cit_pos - min(cit_pos, token_window), cit_pos + min(len(all_tokens) - cit_pos, token_window)):
#             val = cnt - cit_pos
#             if normalize_values:
#                 val = val / float(token_window)
#             all_tokens[cnt][dict_key] = val


def annotateContextTokenDistanceFromCitation(context, cit, token_window=300):
    """
        Adds to each token in the context the distance to the current citation within
        a window of tokens
    """

    all_tokens = []
    for sent in context:
        all_tokens.extend(sent["token_features"])

    # cit_num = getFirstNumberFromString(cit["id"])
    # cit_pos = getCitationTokenPosition(cit_num, all_tokens)
    # cit_index = getFirstCitationTokenIndex(all_tokens)

    cit_index = None
    for index, features in enumerate(all_tokens):
        if features["token_type"] == "c":
            cit_index = index
            break

    if not cit_index:
        # print("Error: couldn't find citation ",cit["id"]) # not really an error, it's just that the citation is outside of the window
        return context

    #
    # if cit_index:
    #     for index, features in enumerate(all_tokens):
    #         if features["token_type"] == "c":
    #             features["dist_cit"] = 0
    #         else:
    #             features["dist_cit"] = cit_index - index
    #
    #         features["abs_dist_cit"] = features["dist_cit"]

    for index in range(cit_index - min(cit_index, token_window),
                       cit_index + min(len(all_tokens) - cit_index, token_window)):
        if all_tokens[index]["token_type"] == "c":
            all_tokens[index]["dist_cit"] = 0

        val = index - cit_index
        all_tokens[index]["dist_cit"] = val
        all_tokens[index]["abs_dist_cit"] = abs(val)
        val = val / float(token_window)
        all_tokens[index]["dist_cit_norm"] = val

    return context


class DocumentFeaturesAnnotator(object):
    """
        Pre-annotates all features in a scidoc to be used for keyword extraction
    """

    def __init__(self, index_name, doc_type="doc", field_name="text", experiment=None):
        self.index_name = index_name
        self.doc_type = doc_type
        self.field_name = field_name
        self.exp = experiment

    def select_sentences(self, doc, window_size=3):
        """
            Selects which sentences to annotate
        """

        to_annotate = [0 for s in range(len(doc.allsentences))]
        for index, s in enumerate(doc.allsentences):
            if len(s.get("citations", [])) > 0:
                for cnt in range(max(index - 2, 0), min(index + 2, len(to_annotate)) + 1):
                    to_annotate[cnt] = True

    def annotate_scidoc(self, doc):
        """
            Adds document-wide features to a scidoc. To add token-level features for contexts, call annotate_context()
        """
        all_parses = {}
        doctext = doc.getFullDocumentText(True, False)
        vector = TfidfVectorizer(decode_error="ignore", stop_words=stopwords)
        matrix = vector.fit_transform([doctext])
        ##        tf = vector.tf_
        ##        tf_scores=dict(zip(vector.get_feature_names(), tf))
        for sent in doc.allsentences:
            # text = replaceCitationTokensForParsing(sent["text"])
            # text = cleanXML(text)
            text = doc.formatTextForExtraction(sent["text"])
            parse = en_nlp.tokenizer(text)

            all_parses[sent["id"]] = parse
            if "token_features" in sent:
                del sent["token_features"]  # remove any existing annotation to avoid confusion

            if "pos_tagged" in sent:
                del sent["pos_tagged"]

        tokens = []
        for parse in all_parses.values():
            for token in parse:
                tokens.append(token.lower_)

        tf_scores = get_tf_scores(tokens)
        df_scores = getElasticTermScores(cp.Corpus.endpoint, doctext, self.index_name, self.field_name)

        # if len(df_scores) > 0:
        #     df_scores = df_scores[self.field_name]["terms"]

        num_docs = getElasticTotalDocs(cp.Corpus.endpoint, self.index_name, self.doc_type)
        doc["precomputed_feature_data"] = {"tf_scores": tf_scores,
                                           "df_scores": df_scores,
                                           "num_docs": num_docs}

        return doc

    def annotate_context(self, sentences, cit, doc):
        """
            Annotate all features for keyword extraction at the context level
        """
        if "precomputed_feature_data" not in doc.data:
            print("!", doc.metadata["guid"])
            return

        tf_scores = doc["precomputed_feature_data"]["tf_scores"]
        df_scores = doc["precomputed_feature_data"]["df_scores"]
        num_docs = doc["precomputed_feature_data"]["num_docs"]

        cit_num = getCitationNumberFromTokenText(cit["id"])

        for sent in sentences:
            token_features = []
            text = doc.formatTextForExtraction(sent["text"])
            text = text.replace(")", " )")
            text = text.replace("(", " (")
            tokens = en_nlp.tokenizer(text)
            sent_text = leaveOneCitationToken(tokens, cit_num)

            parse = en_nlp(sent_text)

            for token in parse:
                text = token.lower_
                tf = tf_scores.get(text, 1)
                df = df_scores.get(text, {}).get("doc_freq", 0)
                tf_idf = sqrt(tf) * (1 + log(num_docs / float(df + 1)))

                features = {"text": token.text,
                            "pos": token.pos,
                            "pos_": token.pos_,
                            "dep": token.dep,
                            "dep_": token.dep_,
                            "ent_type": token.ent_iob,
                            # this is useless for scientific papers. Need a corpus-specific NER
                            "lemma_": token.lemma_ or text,
                            "lemma": token.lemma,
                            "is_punct": token.is_punct,
                            "is_lower": token.is_lower,
                            "like_num": token.like_num,  # Note this will catch 0, million, two, 1,938 and others
                            "is_stop": token.is_stop,
                            "tf": tf,
                            "df": df,
                            "tf_idf": tf_idf,
                            "token_type": "t",  # "t" for normal token, "c" for citation, "a" for author, "p" for punctuation
                            "dist_cit": 0,
                            "abs_dist_cit": 0,
                            }

                if features["text"] == CIT_MARKER:
                    features["token_type"] = "c"
                elif features["text"] == AUTHOR_MARKER:
                    features["token_type"] = "a"
                elif token.is_punct:
                    features["token_type"] = "p"

                if features["token_type"] == "t":
                    features["word_len"] = len(features["text"])
                else:
                    features["word_len"] = 1

                token_features.append(features)

            sent["token_features"] = token_features

        sentences = annotateContextTokenDistanceFromCitation(sentences, cit)

        return sentences

    def save_vocab(self):
        if self.exp:
            import os
            en_nlp.vocab.to_disk(os.path.join(self.exp.get("exp_dir"), "vocab.spacy"))


def main():
    pass


if __name__ == '__main__':
    main()
