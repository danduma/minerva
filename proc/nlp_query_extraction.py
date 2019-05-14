# should be using spacy for everything NLP from now on
from ml.document_features import en_nlp, selectContentWords
from proc.query_extraction import SentenceQueryExtractor, EXTRACTOR_LIST



class FilteredSentenceQueryExtractor(SentenceQueryExtractor):
    def getQueryTextFromSentence(self, sent):
        doc = en_nlp(sent["text"])
        words = selectContentWords(doc)
        text = " ".join(words)
        return text


EXTRACTOR_LIST[ "Sentences_filtered"] = FilteredSentenceQueryExtractor()