# Read XML output from ParsCit into a reference format as used by SciDocJSON
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from BeautifulSoup import BeautifulStoneSoup
from minerva.xmlformats.citation_utils import guessNamesOfPlainTextAuthor

class ParsCitReader:
    def __init__(self):
        pass

    def loadParsCitReference(self, reference):
        """
            Given an XML <citation> node, loads all the relevant values from it.

            Args:
                reference: XML node
            Returns:
                metadata of reference
        """
        metadata={
            "title":"title", # key: the key in the final dict. Value: the XML tag to look for
            "year":"date",
            "volume":"volume",
            "pages":"pages",
            "publisher":"publisher",
            "location":"location",
        }

        for key in metadata:
            # substitute each value string by its value in the XML, if found
            node=reference.find(metadata[key])
            if node:
                text=node.text.strip(".") # just to be annoying. get the actual text of the node
            else:
                text=""
            metadata[key]=text

        for atype in ["journal", "booktitle"]:
            node=reference.find(atype)
            if node:
                metadata["publication"]=node.text.strip(".,: ")
                #TODO: expand this to inproceedings, etc.
                metadata["type"] = atype

        metadata["authors"]=[]
        author_nodes=reference.findAll("author")
        for author_string in [author.text for author in author_nodes]:
            metadata["authors"].append(guessNamesOfPlainTextAuthor(author_string))

        metadata["surnames"]=[author["family"] for author in metadata["authors"]]
        return metadata

    def readReferences(self, root_node):
        """
            Args:
                root_node: BeautifulSoup node from which to search
        """
        references_root=root_node.find("citationlist")
        if not references_root:
            return None

        references=[]
        if root_node:
            for reference in root_node.findAll("citation"):
                references.append(self.loadParsCitReference(reference))
        return references

    def parseParsCitXML(self, xml_string):
        """
            This is meant to load the full output from ParsCit, whichever it may be.
            Currently only reads references.
        """
        soup = BeautifulStoneSoup(xml_string)
        references=self.readReferences(soup)
        # TODO implement reading the rest of the ParsCit/ParsHed tagging
        return references

def main():
    from general_utils import loadFileText
    from reference_formatting import formatReference

    reader=ParsCitReader()
    loaded=reader.parseParsCitXML(loadFileText(r"G:\NLP\ParsCit-win\example.xml"))
    for ref in loaded:
        print formatReference(ref)
    pass

if __name__ == '__main__':
    main()
