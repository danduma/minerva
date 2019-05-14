# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import logging
import db.corpora as cp
from proc import doc_representation
import six

ES_TYPE_DOC = "doc"


def getPrebuildFunctionFromString(function_name):
    """
        Returns a pointer to the doc_representation prebuild function if found
    """
    function = getattr(doc_representation, function_name, None)
    if not function:
        raise ValueError("Unknown function %s" % function_name)
    return function


def prebuildMulti(method_name, parameters, prebuild_function, doc, doctext, guid, overwrite_existing_bows,
                  filter_options, force_rebuild=False):
    """
        Builds multiple BOWs for each document based on multiple parameters.

        :param method_name: string identifying the doc representation method
        :param parameters: parameters to the doc method
        :param prebuild_function: pointer to the function to call to build these BOWs
        :param doc: loaded SciDoc, or None to load it in this function
        :param doctext: loaded SciDoc text, or None to load it in this function
        :param guid: guid of the file being processed
        :param overwrite_existing_bows: if False, only build BOWs that are not in the db already
        :param filter_options: filter options for incoming_link_contexts
    """
    assert isinstance(parameters, list)

    if not overwrite_existing_bows:
        params = cp.Corpus.selectBOWParametersToPrebuild(guid, method_name, parameters)
    else:
        params = parameters
        print("No BOWs available for ", guid, params)

    all_bows = {}
    if len(params) > 0:
        if isinstance(prebuild_function, six.string_types):
            prebuild_function = getPrebuildFunctionFromString(prebuild_function)

        # changed this so doc only gets loaded if absolutely necessary
        retries = 0
        while doc is None:
            try:
                doc = cp.Corpus.loadSciDoc(guid, ignore_errors=["error_match_citation_with_reference"])
                if not doc:
                    raise ValueError("No SciDoc for %s" % guid)
            except Exception as e:
                doc = None
                if retries > 2:
                    # print(e)
                    logging.error("Cannot load SciDoc for %s " % guid)
                    return
                retries += 1

            # TODO make sure this doesn't affect anything. It doesn't make sense to annotate as BOWs are being build. Gotta do that before.
            ##            for annotation in rhetorical_annotations:
            ##                cp.Corpus.annotateDoc(doc, annotation.upper())

        if not doctext:
            doctext = doc.formatTextForExtraction(doc.getFullDocumentText())

        all_bows = prebuild_function(doc, parameters=params, doctext=doctext, filter_options=filter_options,
                                     force_rebuild=force_rebuild)
        ##        print("Saving prebuilt %s BOW for %s" % (method_name,doc["metadata"]["guid"]))

        for param in params:
            param_dict = {"method": method_name, "parameter": param, }
            cp.Corpus.savePrebuiltBOW(
                doc["metadata"]["guid"],
                param_dict,
                all_bows[param])
    return all_bows


def main():
    pass


if __name__ == '__main__':
    main()
