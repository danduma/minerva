# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import logging
import minerva.db.corpora as cp
from minerva.proc import doc_representation

ES_TYPE_DOC="doc"


def prebuildFunction(function_name):
    """
        Returns a pointer to the doc_representation prebuild function if found
    """
    function=getattr(doc_representation, function_name, None)
    if not function:
        raise ValueError("Unknown function %s" % function_name)
    return function


def prebuildMulti(method_name, parameters, function, doc, doctext, guid, force_prebuild, rhetorical_annotations):
    """
        Builds multiple BOWs for each document based on multiple parameters.

        :param method_name: string identifying the doc representation method
        :param parameters: parameters to the doc method
        :param function: pointer to the function to call to build these BOWs
        :param doc: loaded SciDoc, or None to load it in this function
        :param doctext: loaded SciDoc text, or None to load it in this function
        :param guid: guid of the file being processed
        :param force_prebuild: if False, only build BOWs that are not in the db already
    """
    assert isinstance(parameters, list)
    if not force_prebuild:
        params=cp.Corpus.selectBOWParametersToPrebuild(guid,method_name,parameters)
    else:
        params=parameters

    if len(params) > 0:
        if isinstance(function,basestring):
            function=prebuildFunction(function)

        # changed this so doc only gets loaded if absolutely necessary
        if not doc:
            try:
                doc=cp.Corpus.loadSciDoc(guid, ignore_errors=["error_match_citation_with_reference"])
                if not doc:
                    raise ValueError("No SciDoc for %s" % guid)
            except:
                logging.exception("Cannot load SciDoc")

            for annotation in rhetorical_annotations:
                cp.Corpus.annotateDoc(doc, annotation.upper())

            doctext=doc.getFullDocumentText()

        all_bows=function(doc,parameters=params, doctext=doctext)
##        print("Saving prebuilt %s BOW for %s" % (method_name,doc["metadata"]["guid"]))

        for param in params:
            param_dict={"method":method_name, "parameter":param,}
            cp.Corpus.savePrebuiltBOW(
                doc["metadata"]["guid"],
                param_dict,
                all_bows[param])
    return all_bows


def main():

    pass

if __name__ == '__main__':
    main()
