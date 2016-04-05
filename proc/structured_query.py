# StructuredQuery class
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from collections import namedtuple

StructuredToken=namedtuple("StructuredToken",["token", "count","boost","bool","field","distance"])

class StructuredQuery(list):
    """
        Stores a structured query and has methods for de/serializing it
    """
    def __init__(self, data=None):
        """
        """
        if data:
            self.load(data)

    def addToken(self, token, count, boost=1, bool=None, field=None, distance=None):
        """
        """
##        query_token={
##                "token":token,
##                "count":count,
##                "boost":boost,
##                "bool":bool, # +/-
##                "field":field,
##                "distance": distance
##                }

        query_token=StructuredToken(token, count, boost, bool, field, distance)
        self.append(query_token)

    def load(self, data):
        """
            :param data: list of tokens to process, either dicts or tuples
        """
        assert(isinstance(data,list))
        for token in data:
            if isinstance(token,dict):
                self.addToken()
                self.addToken(token["token"],
                              token["count"],
                              token["boost"],
                              token["bool"],
                              token["field"],
                              token["distance"]
                              )
            elif isinstance(token, tuple) or isinstance(token, list):
                self.addToken(**token)
            else:
                raise ValueError("Type not recognized")



def main():
    pass

if __name__ == '__main__':
    main()
