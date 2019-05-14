# All functions related to preprocessing and indexing the bob corpus
#
# Copyright (C) 2014 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
import sqlite3, os, glob
from proc.general_utils import *

import corpus_ingest
from . import aan_metadata

from xmlformats.bobscixml import *
from scidoc import SciDoc

import corpora

# ------------------------------------------------------------------------------
#   Corpus iterating functions
# ------------------------------------------------------------------------------


def generateCSV(filename, index):
    """
        Write to a CSV all the important info about the database
    """

    f=codecs.open(filename,"wb","utf-8", errors="replace")
    line="Filename\tAuthors\tSurnames\tYear\tTitle\tNumRef\tInlinks\tIn-collection_references\tError\tNotes\n"
    f.write(line)

    for doc in index:
        line="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cs(doc["filename"]),
        doc["authors"],
        doc["surnames"],
        cs(doc["year"]),
        cs(doc["title"]),
        cs(str(doc["numref"])),
        cs(str(len(doc["inlinks"]))),
        cs(str(len(doc["outlinks"]))),
        cs(doc["error"]),
        cs(doc["notes"])
        )
##            line=unicode(line,"utf-8","replace")
        f.write(line)

    f.close()


# ------------------------------------------------------------------------------
#   MAIN functions
# ------------------------------------------------------------------------------


def main():
# Convert AAN metadata into a dict[filename]=metadata
    drive="C"
    corpus_ingest.FILE_CONVERSION_FUNCTION=loadSciXML
    root_dir=drive+":\\nlp\\phd\\bob\\"

    metadata_file=root_dir+"fileDB\\db\\aan.pic"
    if not os.path.exists(metadata_file):
        metadata_index=aan_metadata.convertAANmetadata(drive+r":\NLP\PhD\aan\release\2012\acl-metadata.txt",root_dir+"fileDB\\db\\aan.pic")
    else:
        metadata_index=loadPickle(root_dir+"fileDB\\db\\aan.pic")

    corpus_ingest.ingestCorpus(
    root_dir+"inputXML",
    root_dir,
    metadata_index=metadata_index,
    files_to_ignore=loadFileList(root_dir+"fileDB\\db\\files_to_ignore.txt"))

    pass

if __name__ == '__main__':
    main()
