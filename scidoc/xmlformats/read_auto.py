# Automatically detect the type of XML file, use the appropriate reader class
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import re

from base_classes import BaseSciDocXMLReader
from read_jatsxml import JATSXMLReader
from read_paperxml import PaperXMLReader
from read_sapienta_jatsxml import SapientaJATSXMLReader

XML_FORMATS_LIST={
    "JATS":{"regex":r"(//NLM//DTD|JATS|Journal\sArchiving\sand\sInterchange\sDTD)", "reader":JATSXMLReader},
    "paperxml":{"regex":r"paperxml", "reader":PaperXMLReader},
    "SapientaJATSXML": {"regex":"<ebiroot xmlns:z=\"ebistuff\">", "reader": SapientaJATSXMLReader},
##    "SapientaSciXML": {"regex":"paper-structure-annotation.dtd", "reader": None},
##    "SciXML":{"regex":"983hf9aiuhn93ue", "reader": None}
    }

class AutoXMLReader(BaseSciDocXMLReader):
    """
        Detects the file format and instantiates an appropriate reader for it.
    """

    def __init__(self, format=None):
        if format and format not in XML_FORMATS_LIST:
            raise ValueError("Unknown XML format " + format)
        self.format=format

    def selectBestFormat(self, xml):
        """
        """
        match=re.findall("<!DOCTYPE\s(.*?)>",xml[:300],flags=re.IGNORECASE|re.DOTALL)
        if not match:
            print("no DOCTYPE")
            print(xml[:100])
##            print(input_file)
##        if match:
        for format in XML_FORMATS_LIST:
            if re.findall(XML_FORMATS_LIST[format]["regex"], xml, flags=re.IGNORECASE|re.DOTALL):
                return format
        return None

    def read(self, xml, identifier):
        """
            Creates an instance of the right object
        """
        if self.format:
            format_to_use=self.format
        else:
            format_to_use=self.selectBestFormat(xml)
            if not format_to_use:
                raise ValueError("Cannot determine XML file format")

        self.reader=XML_FORMATS_LIST[format_to_use]["reader"]()
        return self.reader.read(xml, identifier)

def inspectFiles():
    """
        Goes file by file printing something about it
    """
    import minerva.db.corpora as cp
    drive="g"
    cp.Corpus.connectCorpus(drive+":\\nlp\\phd\\pmc")

    doc_list=cp.Corpus.selectRandomInputFiles(20,"*.nxml")

    reader=AutoXMLReader()

    for filename in doc_list:
        input_file=cp.Corpus.dir_inputXML+filename
        f=open(input_file)
        lines=f.readlines()
        line="".join(lines[:5])
        format=reader.selectBestFormat(line)
        print(format)


def main():
    inspectFiles()
    pass

if __name__ == '__main__':
    main()
