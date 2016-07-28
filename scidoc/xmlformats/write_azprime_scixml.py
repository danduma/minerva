# Writes AZPrime SciXML
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import re

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from write_scixml import SciXMLWriter, escapeText

from minerva.proc.general_utils import cleanxml

class AZPrimeWriter(SciXMLWriter):
    def __init__(self):
        """
        """
        super(self.__class__, self).__init__()
        self.save_pos_tags=False

    def processSentenceText(self, s, doc):
        """
            This overriden function POS-tags every token in the sentence to spoonfeed it to the AZPrime classifier
        """
        if s.get("pos_tagged","") != "":
            return s["pos_tagged"]

        text=re.sub(r"<cit\sid=\"?(.+?)\"?\s*?/>",r"_CIT_\1_",s["text"], flags=re.DOTALL|re.IGNORECASE)
        text=escapeText(cleanxml(text))
        tokens=word_tokenize(text)
        tags=pos_tag(tokens)
        items=[]
        for t in tags:
            token=t[0]
            pos=t[1]
            if token == "<":
                token="&lt;"
            elif token == ">":
                token="&gt;"
            elif token == "&":
                token="&amp;"

            if token.startswith("_CIT_"):
                items.append("<REF>%s</REF>" % token)
            else:
                if pos in ["''","'"]:
                    items.append("<W C=\"%s\">%s</W>" % (pos,token))
                else:
                    items.append("<W C='%s'>%s</W>" % (pos,token))

        xml_string=" ".join(items)
        if self.save_pos_tags:
            s["pos_tagged"]=xml_string
        return xml_string

def basicTest():
    """
    """
    writer=AZPrimeWriter()
    print(writer.processSentenceText({"text":"Hello Jon, the car is white _CIT_4_."}, {}))

def main():
    basicTest()
    pass

if __name__ == '__main__':
    main()
