# AZ features
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import re, itertools

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from bs4 import BeautifulStoneSoup

from formulaic_patterns import formulaicPattern
from proc.nlp_functions import formatSentenceForIndexing

formPat=formulaicPattern()

#===============================
#       FEATURE EXTRACTION
#===============================
def getSentenceWordsForNgrams(doc,sentence):
    """
        Wrapper method for returning a sentence's text tokens, without citations or
        other inline elements
    """
    text=formatSentenceForIndexing(sentence,no_stemming=True).lower()
    text=re.sub(r"</?cit.*?>","__cit__",text, 0, re.IGNORECASE|re.DOTALL)
    text=re.sub(r"<refauthor.*?</refauthor>","__refauthor__",text, 0, re.IGNORECASE|re.DOTALL)
    text=re.sub(r"</?.*?>"," ",text, 0, re.IGNORECASE|re.DOTALL)
    words=text.split()
##    print words
    return words

def getFeatureBigrams(features,sentence,doc,history=None):
    words=getSentenceWordsForNgrams(doc,sentence)
##    print words
    bigram_finder = BigramCollocationFinder.from_words(words)
    try:
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 20) # similarity function, num features
    except:
        try:
            print("ARGG: error finding bigrams in sentence #",sentence["id"],": ",sentence["text"])
        except:
            print("ARGG: error finding bigrams in sentence: <unicode error>")
        return
    for ngram in itertools.chain(words, bigrams):
        features[ngram]=True

def getFeatureSliceOfDocument(features,sentence,doc, history=None):
    """
        Returns the slice (out of 2) of the document the sentence's section falls into
    """
    NUM_SLICES=2
    features["F_slice_of_document"]=1+((sentence.get("section_counter",0)*NUM_SLICES)/len(doc.allsections))

def getFeatureSentenceLengthMultiples3(features,sentence,doc, history=None):
    """
        The length of a sentence binned in multiples of 3
    """
    words=getSentenceWordsForNgrams(doc,sentence)
    l=(len(words)/3)*3

    features["F_sentence_length_group3"]=l

def getFeatureFirstWords(features,sentence,doc, history=None):
    """
        First 4 words of sentence
    """
    words=getSentenceWordsForNgrams(doc,sentence)

    for word in words[:4]:
        features["F_firstwords_"+word]=True


AZ_all_features={
getFeatureBigrams, # unigrams and bigrams above a certain chisq value
getFeatureSentenceLengthMultiples3, # sentence length, bucketed
getFeatureSliceOfDocument, # slice of the document (out of 2) that sentence occurs in
getFeatureFirstWords # first 4 words added individually as feature
}

AZ_precomputed_features=[
"since_last_header", # num of sentence inside current section
"in_paragraph_counter", # num of sentence inside paragraph
"section_counter", # counter of sections
"in_paragraph_slice", # in which slice of current paragraph
"in_section_slice" # in which slice of current section
]

CFC_all_features={
getFeatureBigrams, # unigrams and bigrams above a certain chisq value
getFeatureSentenceLengthMultiples3, # sentence length, bucketed
getFeatureSliceOfDocument, # slice of the document (out of 2) that sentence occurs in
getFeatureFirstWords # first 4 words added individually as feature
}

CFC_precomputed_features=[
"since_last_header", # num of sentence inside current section
"in_paragraph_counter", # num of sentence inside paragraph
"section_counter", # counter of sections
"in_paragraph_slice", # in which slice of current paragraph
"in_section_slice" # in which slice of current section
]


#===============================
#       FEATURESET BUILDING
#===============================

def prebuildAZFeaturesForDoc(doc):
    """
        Goes through the SciDoc precomputing required features
    """
    def fixEndOfSection(sentences_in_section):
        NUM_SLICES=5
        for index,sentence in enumerate(sentences_in_section):
            sentence["in_section_slice"]=1+((index*NUM_SLICES)/len(sentences_in_section))

    def fixEndOfParagraph(sentences_in_paragraph):
        NUM_SLICES=3
        for index,sentence in enumerate(sentences_in_paragraph):
            sentence["in_paragraph_slice"]=1+((index*NUM_SLICES)/len(sentences_in_paragraph))

    section_counter=0
    since_last_header=1
    in_paragraph_counter=1
    sentences_in_section=[]
    sentences_in_paragraph=[]

    for element in doc.data["content"]:
        if "type" in element:
            if element["type"].lower()=="section":
                section_counter+=1
                fixEndOfSection(sentences_in_section)
                sentences_in_section=[]
                since_last_header=1

            elif element["type"].lower()=="p":
                fixEndOfParagraph(sentences_in_paragraph)
                sentences_in_paragraph=[]
                in_paragraph_counter=1

            elif element["type"].lower()=="s":
                element["section_counter"]=section_counter
                element["since_last_header"]=min(since_last_header,10) # count to a max of X sentences after last heading
                element["in_paragraph_counter"]=min(in_paragraph_counter,10) # count to a max of X sentences in paragraph
                since_last_header+=1
                in_paragraph_counter+=1
                sentences_in_section.append(element)
                sentences_in_paragraph.append(element)

            previous_element=element

    fixEndOfParagraph(sentences_in_paragraph)
    fixEndOfSection(sentences_in_section)


def removePrebuiltAZFeatures(doc):
    """
        Removes the precomputed features added to each sentence dict
    """
    for sentence in doc.allsentences:
        for feature in AZ_precomputed_features:
            if feature in sentence:
                del sentence[feature]

def loadRefAuthorsFromSentence(sentence):
    """
        Converts <refauthor> tags to proper citations
    """
    cfc_citations=[]
    for match in re.findall(r"<refauthor.*?</refauthor>",sentence["text"],re.IGNORECASE):
        soup=BeautifulStoneSoup(match)
        new_cit={"parent_s":sentence["id"]}

        avoid_list={"links"}
        for key in [key[0] for key in soup.attrs if key[0] not in avoid_list]:
            new_cit[key]=soup[key]

        cfc_citations.append(new_cit)
    return cfc_citations

def buildAZFeatureSetForDoc(doc):
    """
        Returns a list of tuples (dict, AZ), where each element represents a sentence
        in the document. The dict contains the features extracted for the sentence,
        AZ is a string representing the Zone (BKG, etc.)
    """

    res=[]

    prebuildAZFeaturesForDoc(doc)

    for sentence in doc.allsentences:
        if "az" in sentence:
            features={}
            for feature in AZ_all_features:
                feature(features,sentence,doc, None)

            for feature in AZ_precomputed_features:
                features["F_"+feature]=sentence[feature]

            formPat.extractFeatures(sentence["text"],features) # formulaic patterns
            formPat.extractFeatures(sentence["text"],features,True) # agent patterns
##            print features

            res.append((features,sentence["az"]))
    return res


def buildCFCFeaturesetForDoc(doc):
    """
        Returns a list of tuples (dict, CF), where each element represents a citation
        in the document. The dict contains the features extracted for the sentence,
        CF is a string representing the Citation Function (BKG, etc.)
    """
    res=[]

    prebuildAZFeaturesForDoc(doc)

    for sentence in doc.allsentences:
        cfc_citations=[doc.citation_by_id[citation] for citation in sentence.get("citations",[])]
        cfc_citations.extend(loadRefAuthorsFromSentence(sentence))

        for citation in cfc_citations:
            if "cfunc" in citation:
                features={}
                for feature in CFC_all_features:
                    feature(features,sentence,doc, None)

                for feature in CFC_precomputed_features:
                    features["F_"+feature]=sentence[feature]

                formPat.extractFeatures(sentence["text"],features) # formulaic patterns
                formPat.extractFeatures(sentence["text"],features,True) # agent patterns

                res.append((features,citation["cfunc"]))
    return res

def addPOSTagsToSentences(document):
    for sentence in document["sentences"]:
        pass

def main():

    pass

if __name__ == '__main__':
    main()
