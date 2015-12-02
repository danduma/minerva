# BaseSciDocXMLReader and BaseSciDocXMLWriter
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from minerva.util.general_utils import loadFileText

class BaseSciDocXMLReader:
    """
        Base class for all SciDoc readers.
    """
    def __init__(self):
        pass

    def readFile(self, filename):
        """
            Load an XML file into a SciDoc.

            Args:
                filename: full path to file to read
            Returns:
                SciDoc instance
        """
        text=loadFileText(filename)
        return self.read(text, filename)

    def read(self, xml, identifier):
        """
            Abstract method that implements the reading. Override in descendant
            classes.

            Args:
                xml: full xml string
                identifier: an identifier for this document, e.g. file name
                        If an actual full path, the path will be removed from it
                        when stored
            Returns:
                SciDoc instance
        """
        raise NotImplementedError

class BaseSciDocXMLWriter:
    def __init__(self):
        pass

    def write(self, doc, filename):
        pass

def main():
    pass

if __name__ == '__main__':
    main()
