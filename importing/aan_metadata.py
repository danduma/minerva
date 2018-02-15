# A bunch of functions to handle the metadata of the ACL Anthology Network (AAN) corpus
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from proc.general_utils import loadFileText
import codecs, re, json


def convertAANmetadata(infile):
    """
        Load strange text file format from AAN, convert to CSV.

        WARNING: breaks backwards compatibility

        Args:
            infile: path to acl-metadata.txt
        Returns:
            returns a dict where [id] = {"authors", "title", etc.}
    """
    alltext=loadFileText(infile)
    filedict={}

    for match in re.finditer(r"id\s\=\s\{(.+?)\}\nauthor\s\=\s\{(.+?)\}\ntitle\s\=\s\{(.+?)\}\nvenue\s\=\s\{(.+?)}\nyear\s\=\s\{(.+?)\}",alltext,re.IGNORECASE):
        fn=match.group(1).lower()
        authors=match.group(2).split(";")
        surnames=[]
        parsed_authors=[]
        for a in authors:
            bits=a.split(",")
            surnames.append(bits[0].strip())
            parsed_authors.append({"given":"".join(bits[1:]).strip(),"family":bits[0].strip()})
        title=match.group(3)
        conference=match.group(4)
        year=match.group(5)
        filedict[fn]={"authors":parsed_authors,"surnames":surnames, "title":title, "conference":conference, "year":year, "corpus_id":fn}
        author_string="["+",".join(authors)+"]"

    return filedict

def convertAANcitations(infile):
    """
        Takes an acl.txt file, parses it to a dict

        Args:
            infile: path to acl.txt
        Returns:
            returns a dict where [id_from] = [list of id_to]
    """
    with open(infile, "r") as f:
        lines=open(infile, "r").readlines()

    citations={}
    for line in lines:
        line=line.lower()
        elements=line.split()
        if len(elements) > 2:
            id_from=elements[0].strip()
            id_to=elements[2].strip()
            citations[id_from]=citations.get(id_from,[])
            if id_to not in citations[id_from]:
                citations[id_from].append(id_to)
    return citations

def convertToJson():
    """
        Convert AAN metadata to JSON
    """
    citations=convertAANcitations("G:\\NLP\\PhD\\aan\\release\\acl_full.txt")
    with open("G:\\NLP\\PhD\\aan\\release\\citations.json", "w") as f:
        json.dump(citations,f)

    metadata=convertAANmetadata("G:\\NLP\\PhD\\aan\\release\\acl-metadata_full.txt")
    with open("G:\\NLP\\PhD\\aan\\release\\metadata.json", "w") as f:
        json.dump(metadata,f)

def makeCSVwithStats():
    """
        Create a .csv file with all the info I care about
    """
    import pandas as pd

    citations=convertAANcitations("G:\\NLP\\PhD\\aan\\release\\acl_full.txt")
    metadata=convertAANmetadata("G:\\NLP\\PhD\\aan\\release\\acl-metadata_full.txt")

    for key in metadata:
        metadata[key]["citations"]=citations.get(key,[])
        metadata[key]["num_in_collection_references"]=len(citations.get(key,[]))

    data=pd.DataFrame(list(metadata.values()))
    data.to_csv("G:\\NLP\\PhD\\aan\\release\\data.csv", encoding="utf-8")

def main():
    makeCSVwithStats()
    pass

if __name__ == '__main__':
    main()
