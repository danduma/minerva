# <description>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


class BaseKeywordExtractor(object):
    """
        Base class for the keyword extractors
    """
    def __init__(self):
        """
        """
        pass

    def train(self, train_set):
        """
        """
        for unique_result in train_set:
            best_kw=selectBestKeywordsForDocument(unique_result)

    def extract(self, doc, cit):
        """
        """
        pass

class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """
        Simple tfidf keyword extractor
    """


    def extract():
        """

        """


def main():
    pass

if __name__ == '__main__':
    main()
