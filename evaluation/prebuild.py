# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import sys, json, datetime, math

import minerva.db.corpora as cp
import minerva.proc.doc_representation as doc_representation
from minerva.proc.general_utils import loadFileText, writeFileText, ensureDirExists

ES_TYPE_DOC="doc"


def prebuildMulti(method_name, parameters, function, guid, doc, doctext, force_prebuild, rhetorical_annotations):
    """
        Builds multiple BOWs for each document based on multiple parameters.

        :param method_name:
    """
    if not force_prebuild:
        params=cp.Corpus.selectBOWParametersToPrebuild(guid,method_name,parameters)
    else:
        params=parameters

    if len(params) > 0:
        # changed this so doc only gets loaded if absolutely necessary
        if not doc:
            doc=cp.Corpus.loadSciDoc(guid)
            # TODO annotation should only happen if there's an option set to that effect

            for annotation in rhetorical_annotations:
                cp.Corpus.annotateDoc(doc, annotation.upper())

            doctext=doc.getFullDocumentText()

        all_bows=function(doc,parameters=params, doctext=doctext)
        print("Saving prebuilt %s BOW for %s" % (method_name,doc["metadata"]["guid"]))

        for param in params:
            param_dict={"method":method_name, "parameter":param,}
            cp.Corpus.savePrebuiltBOW(
                doc["metadata"]["guid"],
                param_dict,
                all_bows[param])
    return [doc,doctext]

def prebuildBOWsForTests(parameters, maxfiles=1000000, FILE_LIST=None, force_prebuild=False, rhetorical_annotations=[]):
    """
        Generates BOWs for each document from its inlinks, stores them in a
        corpus cached file

        :param parameters: list of parameters
        :param maxfiles: max. number of files to process. Simple parameter for debug
        :param force_prebuild: should BOWs be rebuilt even if existing?

    """
    count=0
    if FILE_LIST:
        cp.Corpus.ALL_FILES=FILE_LIST
    else:
        cp.Corpus.ALL_FILES=cp.Corpus.listPapers()

    if rhetorical_annotations != []:
        print("Loading AZ/CFC classifiers")
        cp.Corpus.loadAnnotators()

    print("Prebuilding BOWs for", min(len(cp.Corpus.ALL_FILES),maxfiles), "files...")
    numfiles=min(len(cp.Corpus.ALL_FILES),maxfiles)

    for guid in cp.Corpus.ALL_FILES[:maxfiles]:
        count+=1
        print("Processing ", count, "/", numfiles, " - ", guid)

        doctext=None
        doc=None

        # prebuild BOWs for all entries in the build_db
        for entry in parameters:
            doc,doctext=prebuildMulti(
                entry,
                parameters[entry]["parameters"],
                parameters[entry]["function"],
                guid,
                doc,
                doctext,
                force_prebuild,
                rhetorical_annotations
            )




def main():
    pass

if __name__ == '__main__':
    main()
