# A bunch of functions to handle the metadata of the ACL corpus
#
# Copyright (C) 2013 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from general_utils import *
import codecs, re


def convertAANmetadata(infile, outcsv=None):
    """
        Load strange text file format from AAN, convert to CSV.

        WARNING: breaks backwards compatibility
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
            surnames.append(bits[0])
            parsed_authors.append({"given":bits[1:],"family":bits[0]})
        title=match.group(3)
        conference=match.group(4)
        year=match.group(5)
        filedict[fn]={"authors":parsed_authors,"surnames":surnames, "title":title, "conference":conference, "year":year}
        author_string="["+",".join(authors)+"]"

    if outcsv:
        writeTuplesToCSV(["Filename","Authors","Surnames","Title", "Year","Conference"],alldata,outcsv)

    return filedict

def main():
    convertAANmetadata(r"g:\NLP\PhD\aan\release\2012\acl-metadata.txt",r"g:\NLP\PhD\bob\fileDB\db\aan.pic",r"C:\NLP\PhD\bob\aan.csv")
    pass

if __name__ == '__main__':
    main()
