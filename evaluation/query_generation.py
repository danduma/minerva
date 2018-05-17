# Generate a query from each resolvable citation for testing purposes
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import math, os, random, json
from copy import deepcopy
from collections import defaultdict, OrderedDict

import db.corpora as cp
from proc.results_logging import ProgressIndicator
# from tqdm import tqdm
from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from proc.doc_representation import findCitationInFullTextXML, findCitationInFullTextUnderscores
from scidoc.citation_utils import getAuthorNamesAsOneString
from .base_pipeline import getDictOfTestingMethods
from importing.fix_citations import fixDocRemovedCitations

GLOBAL_FILE_COUNTER = 0


class QueryGenerator(object):
    """
        Loops over the testing files, generating queries for each citation context.

        The actual query generation happens in the QueryExtractor classes
    """

    def __init__(self):
        self.citations_processed = 0
        self.options = {"jump_paragraphs": True,
                        "add_full_sentences": False,
                        "force_regenerate_resolvable_citations": False}
        self.exclude_sources_targets={}
        self.precomputed_queries=[]

    def saveAllQueries(self):
        """
            Dumps all precomputed queries to disk.
        """
        json.dump(self.precomputed_queries, open(os.path.join(self.exp["exp_dir"], "precomputed_queries.json"), "w"))
        queries_by = {}
        annot_types = ["az", "cfc", "csc_type", "csc_adv", "csc_nov"]
        for annot_type in annot_types:
            queries_by[annot_type] = defaultdict(lambda: [])

        if self.exp.get("random_zoning", False):
            queries_by["rz7"] = defaultdict(lambda: [])
            queries_by["rz11"] = defaultdict(lambda: [])
        if self.exp.get("use_rhetorical_annotation", False):
            for precomputed_query in self.precomputed_queries:
                for annot_type in annot_types:
                    if precomputed_query.get(annot_type, "") != "":
                        queries_by[annot_type][precomputed_query[annot_type]].append(precomputed_query)

                if self.exp.get("random_zoning", False):
                    queries_by["rz7"][random.choice(RANDOM_ZONES_7)].append(precomputed_query)
                    queries_by["rz11"][random.choice(RANDOM_ZONES_11)].append(precomputed_query)

        json.dump(self.files_dict,
                  open(self.exp["exp_dir"] + self.exp.get("files_dict_filename", "files_dict.json"), "w"))
        if self.exp.get("use_rhetorical_annotation", False):
            for annot_type in annot_types:
                json.dump(queries_by[annot_type],
                          open(os.path.join(self.exp["exp_dir"], "queries_by_%s.json" % annot_type), "w"))

        if self.exp.get("random_zoning", False):
            json.dump(queries_by["rz7"], open(os.path.join(self.exp["exp_dir"], "queries_by_rz7.json"), "w"))
            json.dump(queries_by["rz11"], open(os.path.join(self.exp["exp_dir"], "queries_by_rz11.json"), "w"))

        json.dump(self.exclude_sources_targets, open(os.path.join(self.exp["exp_dir"], "exclude_sources_targets.json"), "w"))

    def loadDocAndResolvableCitations(self, guid):
        """
            Deals with all the loading of the SciDoc.

            Throws ValueError if cannot load doc.

            :param guid: GUID of the doc to load
            :returns (doc, doctext, precomputed_file)
            :rtype: tuple
        """
        doc = cp.Corpus.loadSciDoc(guid, ignore_errors=[
            "error_match_citation_with_reference"])  # load the SciDoc JSON from the cp.Corpus
        if not doc:
            raise ValueError("ERROR: Couldn't load pickled doc: %s" % guid)

        doctext = doc.formatTextForExtraction(doc.getFullDocumentText())  # store a plain text representation

        # load the citations in the document that are resolvable, or generate if necessary
        citations_data = cp.Corpus.loadOrGenerateResolvableCitations(doc,
                                                                     filter_options=self.exp[
                                                                         "filter_options_resolvable"],
                                                                     force_recompute=self.options.get(
                                                                         "force_regenerate_resolvable_citations",
                                                                         False))
        return [doc, doctext, citations_data]

    def makeIndividualDictForEachQuery(self, cit_dict, parent_s, queries, precomputed_query, doc_position):
        """
            Expands and flattens the generated queries: essentially it makes
            redundant copies of all values in the dicts so each dict can stand on its own

            :param doc: the SciDoc
            :param parent_s: the Sentence containing the citation for which we're doing this
            :param queries: the returned queries, as generated by extractMulti()
            :param precomputed_query: pre-filled dict with other values to be incorporated in the final dict
        """
        base_dict = {"az": parent_s.get("az", "").strip(),
                     "cfc": cit_dict.get("cfunc", ""),
                     "csc_type": parent_s.get("csc_type", "").strip(),
                     "csc_adv": parent_s.get("csc_adv", ""),
                     "csc_nov": parent_s.get("csc_nov", ""),
                     }

        generated_queries = []

        # for every generated query for this context
        for qmethod in queries:
            this_query = deepcopy(precomputed_query)
            this_query["query_method"] = qmethod
            this_query["query_text"] = queries[qmethod].get("text", "")
            this_query["structured_query"] = queries[qmethod]["structured_query"]
            this_query["doc_position"] = doc_position
            for key in base_dict:
                this_query[key] = base_dict[key]

            # for every method used for extracting BOWs
            generated_queries.append(this_query)

        return generated_queries

    def generateQueriesForCitation(self, citation, doc, doctext, precomputed_query):
        """
            Generate all queries for a resolvable citation.

            :param citation: resolvable citation {cit, match_guid}
            :type citation:dict
            :param doc: SciDoc
            :type doc:SciDoc
            :param doctext: full document text as rendered
            :type doctext:basestring
            :param precomputed_query: pre-filled dict
            :type precomputed_query:dict
        """
        queries = {}
        generated_queries = []

        match = findCitationInFullTextXML(citation["cit"], doctext)
        if not match:
            match = findCitationInFullTextUnderscores(citation["cit"], doctext)
        if not match:
            print("Weird! can't find citation in text!", citation["cit"])
            print("Fixing document ", doc["metadata"]["guid"])
            fixDocRemovedCitations(doc)
            doctext = doc.formatTextForExtraction(doc.getFullDocumentText())
            match = findCitationInFullTextXML(citation["cit"], doctext)
            if not match:
                match = findCitationInFullTextUnderscores(citation["cit"], doctext)
            if not match:
                print("Failed to fix for this citation")
                return generated_queries
            else:
                cp.Corpus.saveSciDoc(doc)

        # this is where we are in the document
        position = match.start()
        doc_position = math.floor((position / float(len(doctext))) * self.exp.get("numchunks", 10)) + 1

        # generate all the queries from the contexts
        for method_name in self.exp["qmethods"]:
            method = self.exp["qmethods"][method_name]

            params = {
                "method_name": method_name,
                "match_start": match.start(),
                "match_end": match.end(),
                "doctext": doctext,
                "docfrom": doc,
                "cit": citation["cit"],
                "dict_key": "text",
                "parameters": method["parameters"],
                "separate_by_tag": method.get("separate_by_tag", ""),
                "options": self.options
            }

            all_queries = method["extractor"].extractMulti(params)
            for query in all_queries:
                queries[query["query_method_id"]] = query

        parent_s = doc.element_by_id[citation["cit"]["parent_s"]]
        cit_dict = doc.citation_by_id[citation["cit"]["id"]]
        generated_queries = self.makeIndividualDictForEachQuery(cit_dict, parent_s, queries, precomputed_query,
                                                                doc_position)

        return generated_queries

    def processOneFile(self, guid):
        """
            Precompute the queries for a single test document

            :param guid: GUID of the document to process
            :type guid: string
        """
        doc, doctext, citations_data = self.loadDocAndResolvableCitations(guid)

        resolvable = citations_data["resolvable"]  # list of resolvable citations
        in_collection_references = citations_data["outlinks"]  # list of cited documents (refereces)

        # TODO do I really need all this information? Recomputed as well?
        num_in_collection_references = len(in_collection_references)
        ##            print ("Resolvable citations:",len(resolvable), "In-collection references:",num_in_collection_references)

        precomputed_file = {"guid": guid, "in_collection_references": num_in_collection_references,
                            "resolvable_citations": len(resolvable), }

        if not self.exp["full_corpus"]:
            precomputed_file["tfidf_models"] = []
            for method in self.all_doc_methods:
                # get the actual dir for each retrieval method, depending on whether full_corpus or not
                actual_dir = cp.Corpus.getRetrievalIndexPath(guid, self.all_doc_methods[method]["index_filename"],
                                                             self.exp["full_corpus"])
                precomputed_file["tfidf_models"].append({"method": method, "actual_dir": actual_dir})

        self.files_dict[guid] = precomputed_file

        # FIXME: INSERT KEA/TEXTRANK HERE insert here ==================

        for citation in resolvable:
            precomputed_query = {"file_guid": guid,
                                 "citation_id": citation["cit"]["id"],
                                 "match_guids": citation["match_guids"],
                                 "citation_multi": citation["cit"].get("multi", 1),
                                 }
            for match_guid in citation["match_guids"]:
                self.exclude_sources_targets[match_guid] = self.exclude_sources_targets.get(match_guid, {"authors":[], "guids_from":[]})
                authors=getAuthorNamesAsOneString(doc.metadata)
                self.exclude_sources_targets[match_guid]["authors"].append(authors[0])
                self.exclude_sources_targets[match_guid]["guids_from"].append(doc.metadata["guid"])

            if self.citations_processed >= self.exp.get("max_queries_generated",10000000):
                return 
            self.precomputed_queries.extend(self.generateQueriesForCitation(citation,
                                                                            doc,
                                                                            doctext,
                                                                            precomputed_query))
            self.citations_processed += 1

    def generateQueries(self, exp):
        """
            Precompute all queries for all annotated citation contexts

            :param exp: experiment dict with all options
            :type exp: dict
        """
        self.exp = exp
        print("Generating queries...")
        logger = ProgressIndicator(True, numitems=len(exp["test_files"]))  # init all the logging/counting
        logger.numchunks = exp.get("numchunks", 10)

        cp.Corpus.loadAnnotators(self.exp.get("rhetorical_annotations", []))

        # convert nested dict to flat dict where each method includes its parameters in the name
        self.all_doc_methods = getDictOfTestingMethods(exp["doc_methods"])

        self.precomputed_queries = []
        self.files_dict = OrderedDict()

        ##        if exp["full_corpus"]:
        ##            files_dict["ALL_FILES"]={}
        ##            files_dict["ALL_FILES"]["doc_methods"]=all_doc_methods
        ##            files_dict["ALL_FILES"]["tfidf_models"]=[]
        ##            for method in all_doc_methods:
        ##                actual_dir=cp.Corpus.getRetrievalIndexPath("ALL_FILES",all_doc_methods[method]["index_filename"],exp["full_corpus"])
        ##                files_dict["ALL_FILES"]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

        # ===================================
        # MAIN LOOP over all testing files
        # ===================================
        for guid in exp["test_files"]:
            try:
                self.processOneFile(guid)
            except ValueError:
                print("Can't load SciDoc ", guid)
                continue

            if self.citations_processed >= self.exp.get("max_queries_generated",10000000):
                break
            logger.showProgressReport(guid)  # prints out info on how it's going

        self.saveAllQueries()
        print("Precomputed queries saved.")


# ===============================================================================
#  functions to measure score
# ===============================================================================

##def analyticalRandomChanceMRR(numinlinks):
##    """
##        Returns the MRR score based on analytical random chance
##    """
##    res=0
##    for i in range(numinlinks):
##        res+=(1/float(numinlinks))*(1/float(i+1))
##    return res


##def addNewWindowQueryMethod(queries, name, method, match, doctext):
##    """
##        Runs a multi query generation function, adds all results with procedural
##        identifier to queries dict
##    """
##    all_queries= method["function"](match, doctext, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
##    for cnt, p_size in enumerate(method["parameters"]):
##        method_name=name+str(p_size[0])+"_"+str(p_size[1])
##        queries[method_name]=all_queries[cnt]
##
##def addNewSentenceQueryMethod(queries, name, method, docfrom, cit, param):
##    """
##        Runs a multi query generation function, adds all results with procedural
##        identifier to queries dict
##    """
####    docfrom, cit, param, separate_by_tag=None, dict_key="text")
##
##    all_queries= method["function"](docfrom, cit, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
##    for cnt, param in enumerate(method["parameters"]):
##        method_name=name+"_"+param
##        queries[method_name]=all_queries[cnt]["text"]

def createExplainQueriesByAZ(retrieval_results_filename="prr_all_ilc_AZ_paragraph_ALL.json"):
    """
        Creates _by_az and _by_cfc files from precomputed_retrieval_results
    """
    retrieval_results = json.load(open(cp.Corpus.paths.prebuiltBOWs + retrieval_results_filename, "r"))
    retrieval_results_by_az = {zone: [] for zone in AZ_ZONES_LIST}

    for retrieval_result in retrieval_results:
        retrieval_results_by_az[retrieval_result["az"]].append(retrieval_result)
    ##        retrieval_results_by_cfc[retrieval_result["query"]["cfc"]].append(retrieval_result)

    json.dump(retrieval_results_by_az, open(cp.Corpus.paths.prebuiltBOWs + "retrieval_results_by_az.json", "w"))


##    json.dump(retrieval_results_by_cfc,open(cp.Corpus.paths.prebuiltBOWs+"retrieval_results_by_cfc.json","w"))


def fixEndOfFile():
    """
        Fixing bad JSON or dumping
    """
    exp = {}
    exp["exp_dir"] = r"C:\NLP\PhD\bob\experiments\w20_csc_fa_w0135" + os.sep

    files = {}
    AZ_LIST = [zone for zone in AZ_ZONES_LIST if zone != "OWN"]
    print(AZ_LIST)

    for div in AZ_LIST:
        files["AZ_" + div] = open(exp["exp_dir"] + "prr_AZ_" + div + ".json", "r+")

    ##    for div in CORESC_LIST:
    ##        files["CSC_"+div]=open(exp["exp_dir"]+"prr_CSC_"+div+".json","r+")

    files["ALL"] = open(exp["exp_dir"] + "prr_ALL.json", "r+")

    for div in AZ_LIST:
        files["AZ_" + div].seek(-1, os.SEEK_END)
        files["AZ_" + div].write("]")


##    for div in CORESC_LIST:
##        files["CSC_"+div].seek(-1,os.SEEK_END)
##        files["CSC_"+div].write("]")

##    files["ALL"].seek(-1,os.SEEK_END)
##    files["ALL"].write("]")


def main():
    pass


if __name__ == '__main__':
    main()
