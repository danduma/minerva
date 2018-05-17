# Corpus class. Deals with a database of scientific papers.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import

import os
import re
import unicodedata
import uuid

import six

from proc.general_utils import (AttributeDict, normalizeTitle, removeSymbols)
from scidoc.citation_utils import getAuthorNamesAsOneString, isSameFirstAuthor, getOverlappingAuthors
from scidoc.scidoc import SciDoc

TABLE_PAPERS = "papers"
TABLE_SCIDOCS = "scidocs"
TABLE_CACHE = "cache"
TABLE_AUTHORS = "authors"
TABLE_LINKS = "links"
TABLE_VENUES = "venues"
TABLE_MISSING_REFERENCES = "missing_references"

CACHE_RESOLVABLE = "resolvable"
CACHE_BOW = "bow"

BUILDING_STATS = {}


class BaseReferenceMatcher(object):
    """
        Base class for ReferenceMatcher. This class provides the matchReference
        function, which must find the right reference in the corpus
    """

    def __init__(self, corpus):
        """
        """
        self.corpus = corpus

    def matchReference(self, ref, doc=None):
        """
        """
        return None

    def matchAuthor(self, author):
        """
            Returns the matching author based on name and surname

            :param author: author dict ["given","family"]
        """
        self.corpus.checkConnectedToDB()

        # try matching by ID first
        if author.get("author_id", "") not in ["", None]:
            author_record = self.corpus.getAuthorByField("author_id", author["author_id"])
            if author_record:
                return author_record

        for field in ["given", "middle", "family"]:
            if field in author:
                author[field] = removeSymbols(author[field])

        query = "SELECT author FROM authors WHERE author.given=\"%s\" and author.family=\"%s\"" % (
            author["given"], author["family"])
        if author.get("middle", "") != "":
            query += " and author.middle=\"%s\"" % author["middle"]

        matches = self.corpus.SQLQuery(query)

        if matches:
            return matches[0]
        else:
            return None

        ##        rows=self.corpus.listFieldByField("author_id","norm_title",hash, table="authors")
        ##
        ##        for row in rows:
        ##            doc_meta=row
        ####            doc_meta=json.loads(row[0]) # load metadata dict
        ##            if len(doc_meta["surnames"]) > 0:
        ##                for a1 in doc_meta["surnames"]:
        ##                    for a2 in ref["surnames"]:
        ##                        if a1 and a2 and a1.lower() == a2.lower():
        ##                            # essentially, if ANY surname matches
        ##                            return doc_meta
        return None


class DefaultReferenceMatcher(BaseReferenceMatcher):
    """
        Default ReferenceMatcher. Matches a reference with its paper in the Corpus
    """

    def matchReference(self, ref, doc=None):
        """
            Returns the matching document in the db based on the guid, corpus_id,
            title and surnames of authors

            :param ref: reference dict
            :aram doc: SciDoc (optional). Only here to enable decendant classes to use
        """
        self.corpus.checkConnectedToDB()

        # try matching by ID first
        for id_type in [("doi", "doi"), ("pmid", "corpus_id"), ("corpus_id", "corpus_id"), ("guid", "guid")]:
            if ref.get(id_type[0], "") not in ["", None]:
                # print(id_type[1], ref[id_type[0]])
                value = ref[id_type[0]]
                # there's an error in the parsing if there's a quotation mark in there
                if "\"" in value or "\\" in value:
                    continue
                try:
                    doc_meta = self.corpus.getMetadataByField(id_type[1], value)
                except Exception as e:
                    print(e)
                    doc_meta = None
                if doc_meta:
                    return doc_meta

        # if it can't be matched by id
        norm_title = normalizeTitle(ref["title"])

        if not isinstance(norm_title, six.text_type):
            norm_title = six.text_type(norm_title, errors="ignore")

        rows = self.corpus.listFieldByField("metadata", "norm_title", norm_title)

        for row in rows:
            doc_meta = row
            ##            doc_meta=json.loads(row[0]) # load metadata dict
            if len(doc_meta["surnames"]) > 0:
                for a1 in doc_meta["surnames"]:
                    for a2 in ref["surnames"]:
                        if a1 and a2 and a1.lower() == a2.lower():
                            # essentially, if ANY surname matches
                            return doc_meta
        return None


def shouldIgnoreCitation(source_metadata, target_metadata, filter_options):
    """
    If the citation is outside the date range we care about or there are more overlapping authors
    between source and target than permitted, returns True.

    :param source_metadata:
    :param target_metadata:
    :param exclude_same_first_author:
    :param max_overlapping_authors:
    :param filter_options: options for filtering, including max year for paper, same authors
    :return:xw
    """
    max_overlapping_authors = filter_options.get("max_overlapping_authors")
    exclude_same_first_author = filter_options.get("exclude_same_first_author", False)

    BUILDING_STATS["bows_checked"] = BUILDING_STATS.get("bows_checked", 0) + 1

    if int(source_metadata["year"]) > filter_options.get("max_year", 9999):
        BUILDING_STATS["self_citation"] = BUILDING_STATS.get("self_citation", 0) + 1
        print("Ignoring ", source_metadata["guid"], " because of YEAR cut-off", source_metadata["year"], " > ",
              filter_options.get("max_year"))
        return True

    if exclude_same_first_author or max_overlapping_authors is not None:
        authors1 = getAuthorNamesAsOneString(source_metadata)
        authors2 = getAuthorNamesAsOneString(target_metadata)
        if max_overlapping_authors is not None and getOverlappingAuthors(
                authors1,
                authors2) > max_overlapping_authors:
            BUILDING_STATS["author_overlap"] = BUILDING_STATS.get("author_overlap", 0) + 1
            print("Ignoring ", source_metadata["guid"], " because of author overlap")
            return True

        if exclude_same_first_author and isSameFirstAuthor(authors1, authors2):
            BUILDING_STATS["self_citation"] = BUILDING_STATS.get("self_citation", 0) + 1
            print("Ignoring", source_metadata["guid"], "self citation", authors1[0])
            return True

    return False


class BaseCorpus(object):
    """
        Base Corpus class: implements most corpus access functions, does not
        implement functions saving and loading; these must be overriden by
        descendant classes
    """

    def __init__(self):
        """
            Basic creator
        """
        # annotate files with AZ, CSC, etc.
        self.annotators = {}
        # pretty-print saved SciDoc files?
        self.saveDocIndent = None
        self.setPaths("")
        self.matcher = DefaultReferenceMatcher(self)
        # any extra data to hold
        self.global_counters = {}
        self.AUTO_ADD_AUTHORS = True
        self.query_filter = ""

    def setPaths(self, root_dir):
        """
            Basic setPaths, mostly here for cachedJson
        """
        self.ROOT_DIR = root_dir
        self.paths = AttributeDict()
        self.paths.inputXML = self.ROOT_DIR + "inputXML" + os.sep
        self.paths.fileDB = self.ROOT_DIR + "fileDB" + os.sep
        self.paths.inputXML = self.ROOT_DIR + "inputXML" + os.sep
        self.paths.experiments = self.ROOT_DIR + "experiments" + os.sep
        self.paths.output = self.ROOT_DIR + "output" + os.sep

    def connectCorpus(self, base_directory, initializing_corpus=False, suppress_error=False):
        """
            If DB has been created, connect to it. If not, initialize it first.

            :param base_directory: root dir of this corpus
            :param initializing_corpus: if True, create DB and directories
            :param suppress_error: if true, db doesn't complain if it's connected already
        """
        raise NotImplementedError

    def loadAnnotators(self, annotators=["AZ", "CFC"]):
        """
            Loads annotators (classifiers), either as parameter or by default
        """
        for annotator in annotators:
            if annotator == "AZ":
                from az.az_cfc_classification import AZannotator
                self.annotators[annotator] = AZannotator("trained_az_classifier.pickle")
            if annotator == "CFC":
                from az.az_cfc_classification import CFCannotator
                self.annotators[annotator] = CFCannotator("trained_cfc_classifier.pickle")

    def annotateDoc(self, doc, annotate_with=None):
        """
            Shortcut to get Corpus object to annotate file with one or more annotators
        """
        if not annotate_with:
            annotate_with = list(self.annotators.keys())

        for annotator in annotate_with:
            assert (annotator in self.annotators)
            self.annotators[annotator].annotateDoc(doc)

    def generateGUID(self, metadata=None):
        """
            Generates a UUID version 4 (random).
        """
        hex_string = str(uuid.uuid4())
        return hex_string

    def generateAuthorID(self, author=None):
        """
            Generates a UUID version 4 (random).
        """
        hex_string = str(uuid.uuid4())
        return hex_string

    def generateDeterministicGUID(self, metadata):
        """
            Generates a UUID version 5. It is based on the authors' surnames and
            title of the document

            >>> print LocalCorpus().generateGUID({"surnames":[u"Jones",u"Smith"], "year": 2005, "norm_title": u"a simple title"})
            296da00b-5fb9-5800-8b5f-f50ef951afc4
        """
        if metadata.get("norm_title", "") == "" or metadata.get("surnames", []) == []:
            raise ValueError("No norm_title or no surnames!")

        uuid_string = " ".join(metadata["surnames"]).lower().strip()
        uuid_string += metadata["norm_title"]
        uuid_string = unicodedata.normalize("NFKC", uuid_string).encode("utf-8")
        hex_string = str(uuid.uuid5(uuid.NAMESPACE_DNS, uuid_string))
        return hex_string

    def updateAllPapersMetadataOutlinks(self):
        """
            Goes through the papers table, updating the outlinks in the metadata
            field with the actual in-collection references.

            Returns a list of guids of documents that have experienced changes,
            so that indeces can be rebuilt accordingly if neccesary
        """

        res = []
        for paper_guid in self.listPapers():
            metadata = self.getMetadataByGUID(paper_guid)
            old_outlinks = metadata["outlinks"]
            doc = self.loadSciDoc(metadata["guid"])
            new_outlinks = self.listDocInCollectionReferences(doc)

            if set(new_outlinks) != set(old_outlinks):
                res.append(paper_guid)
            metadata["outlinks"] = new_outlinks
            self.updatePaper(metadata)

        return res

    def listDocInCollectionReferences(self, doc):
        """
            Returns a list of guids of all the in-collection references of a doc
        """
        res = []
        for ref in doc["references"]:
            match = self.matcher.matchReference(ref)
            if match:
                res.append(match["guid"])
        return res

    def tagAllReferencesAsInCollectionOrNot(self, doc):
        """
            For each reference in a doc, it adds a boolean "in_collection"
        """
        for ref in doc["references"]:
            match = self.matcher.matchReference(ref)
            ref["in_collection"] = True if match else False

    def listInCollectionReferencesOfList(self, guid_list):
        """
            Will return all in-collection references of all the files in the list
        """
        res = []
        assert isinstance(guid_list, list)
        for guid in guid_list:
            res.extend(self.getMetadataByGUID(guid)["outlinks"])
        res = list(set(res))
        return res

    def matchAllReferences(self, doc):
        """
            Matches each reference in a SciDoc with its guid in the collection,
            if found.

            :type doc:SciDoc
        """
        for ref in doc.references:
            ref["guid"] = None
            match = self.matcher.matchReference(ref, doc)
            if match:
                ref["guid"] = match["guid"]

    def selectDocResolvableCitations(self, doc, filter_options={}):
        """
            Returns a list of {"cit","match"} with the citations in the text and
            their matching document in the index

            WARNING! Also removes inline citations that aren't resolvable

            :param doc: SciDoc
            :type doc: SciDoc
            :param filter_options: max_year: Maximum year to select from. exclude_same_first_author: if target document has same first author as ilc_source document
            :returns: tuple of (resolvable_citations, outlinks, missing_references)
        """

        resolvable = []
        unique_outlinks = {}

        sents_with_multicitations = []
        missing_references = []
        max_year= filter_options.get("max_year", 9999)

        cit_to_guid = {}

        for ref in doc["references"]:
            match = self.matcher.matchReference(ref, doc)
            if match and not shouldIgnoreCitation(doc.metadata, match, filter_options):
                ref["guid"] = match["guid"]
                for cit_id in ref.get("citations", []):
                    # resolvable.append({"cit": doc.citation_by_id[cit_id], "match_guid": match["guid"]})
                    cit_to_guid[cit_id] = match["guid"]
                    unique_outlinks[match["guid"]] = unique_outlinks.get(match["guid"], 0) + 1
            else:
                missing_references.append(ref)
                # remove citation from sentence
                for cit_id in ref.get("citations", []):
                    cit = doc.citation_by_id[cit_id]
                    sent = doc.element_by_id[cit["parent_s"]]
                    sent["text"] = re.sub(r"<cit\sid=.?" + str(cit["id"]) + ".{0,3}/>", "", sent["text"], 0,
                                          re.IGNORECASE)
                    sents_with_multicitations.append(sent)

        for sent in sents_with_multicitations:
            # deal with many citations within characters of each other: make them know they are a cluster
            doc.countMultiCitations(sent)

        added_cits = set()
        resolvable = []
        for cit_id in cit_to_guid:
            if cit_id in added_cits:
                continue

            cit = doc.citation_by_id[cit_id]
            entry = {"cit": cit}
            if cit.get("multi", 1) > 1:

                match_guids = []
                for group_cit_id in cit["group"]:
                    if group_cit_id in cit_to_guid:
                        match_guids.append(cit_to_guid[group_cit_id])

                entry["match_guids"] = match_guids
                added_cits.update(set(cit["group"]))
            else:
                entry["match_guids"] = [cit_to_guid[cit_id]]
                added_cits.add(cit_id)
            resolvable.append(entry)

        return [resolvable, unique_outlinks, missing_references]

    def selectBOWParametersToPrebuild(self, guid, method, parameters):
        """
            Returns a list of parameters to build BOWs for, which may be empty,
            by removing all those that already have a prebuiltBOW

            Delete the files if you want them prebuilt again
        """
        newparam = [param for param in parameters if not
        self.cachedJsonExists(CACHE_BOW, guid,
                              {"method": method, "parameter": param}
                              )]
        return newparam

    def savePrebuiltBOW(self, guid, params, bow):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path = self.cachedDataIDString(CACHE_BOW, guid, params)
        self.saveCachedJson(path, bow)

    def loadPrebuiltBOW(self, guid, params):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path = self.cachedDataIDString(CACHE_BOW, guid, params)
        return self.loadCachedJson(path)

    def saveResolvableCitations(self, guid, citations_data):
        """
            Wrapper function to make the code cleaner
        """
        guid = guid.lower()
        self.saveCachedJson(self.cachedDataIDString(CACHE_RESOLVABLE, guid), citations_data)

    def loadResolvableCitations(self, guid):
        """
            Wrapper function to make the code cleaner
        """
        guid = guid.lower()
        return self.loadCachedJson(self.cachedDataIDString(CACHE_RESOLVABLE, guid))

    def generateResolvableCitations(self, doc_file, save_cache=True, filter_options={}):
        """
            Generate and optionally save the resolvable citations for a document.
        """
        resolvable, outlinks, missing_references = self.selectDocResolvableCitations(doc_file, filter_options)
        citations_data = {"resolvable": resolvable, "outlinks": outlinks, "missing_references": missing_references}
        if save_cache:
            self.saveResolvableCitations(doc_file["metadata"]["guid"], citations_data)
        return citations_data

    def loadOrGenerateResolvableCitations(self, doc, filter_options={}, force_recompute=False):
        """
            Tries to open pre-computed resolvable citations and outlinks,
            generates and saves them if not found

            Args:
                :param doc: SciDoc
                :param max_year:
        """

        if force_recompute:
            self.bulkDelete([self.cachedDataIDString(CACHE_RESOLVABLE, doc["metadata"]["guid"])], "cache")

        if force_recompute or not self.cachedJsonExists(CACHE_RESOLVABLE, doc["metadata"]["guid"]):
            citations_data = self.generateResolvableCitations(doc, save_cache=True, filter_options=filter_options)
        else:
            citations_data = self.loadResolvableCitations(doc["metadata"]["guid"])
            if not citations_data:
                citations_data = self.generateResolvableCitations(doc, save_cache=True,
                                                                  filter_options=filter_options)
            #
            #     citations_data = self.selectDocResolvableCitations(doc_file, filter_options)

        return citations_data

    def checkConnectedToDB(self):
        """
            Asserts that the DB is connected
        """
        assert self.connectedToDB(), "Not connected to DB!"

    # -------------------------------------------------------------------------------
    #   Subclass-specific functions from here on
    # -------------------------------------------------------------------------------
    def connectedToDB(self):
        """
            returns True if connected to DB, False otherwise
        """
        raise NotImplementedError

    def cachedJsonExists(self, type, guid, params=None):
        """
        """
        raise NotImplementedError

    def getRetrievalIndexPath(self, guid, index_filename, full_corpus=False):
        """
            Returns the path to the Lucene index for a test file in
            the corpus

            if full_corpus is True, this is the general index for that method
                else
            when using Citation Resolution (resolving only from the
            references at the bottom of a paper) it is the specific index for
            that file guid
        """
        raise NotImplementedError

    def cachedDataIDString(self, type, guid, params=None):
        """
            Returns a string that can be used as either a path or resource ID
        """
        if type == CACHE_RESOLVABLE:
            return type + "_" + guid
        if type == CACHE_BOW:
            return type + "_" + guid + "_" + params["method"] + "_" + six.text_type(params["parameter"])

    def saveCachedJson(self, path, data):
        """
            Save anything as JSON
        """
        raise NotImplementedError

    def loadCachedJson(self, path):
        """
            Load precomputed JSON
        """
        raise NotImplementedError

    def deleteCachedJson(self, path):
        """
            Delete precomputed JSON
        """
        self.bulkDelete([path], TABLE_CACHE)

    def bulkDelete(self, id_list, table=TABLE_PAPERS):
        """
            Deletes all entries in id_list from the given table that match on id.
        """
        raise NotImplementedError

    def loadSciDoc(self, guid):
        """
            If a SciDocJSON file exists for guid, it returns it, otherwise None
        """
        raise NotImplementedError

    def saveSciDoc(self, doc):
        """
            Saves the document as JSON using the SciDoc's saveToFile method,
            name is generated from its GUID + .json
        """
        raise NotImplementedError

    def connectToDB(self, suppress_error=False):
        raise NotImplementedError

    def createAndInitializeDatabase(self):
        """
            Ensures that the directory structure is in place and creates
            the SQLite database and tables
        """
        raise NotImplementedError

    def getRecord(self, id, table=TABLE_PAPERS):
        """
            Abstracts over getting data from a row in the db. Returns all the
            fields of the record for one type of table.

            :param id: id of the record
            :param table: table alias, e.g. [TABLE_PAPERS, TABLE_SCIDOCS]
        """
        raise NotImplementedError

    def getRecordField(self, id, table=TABLE_PAPERS):
        """
            Abstracts over getting data from a row in the db. Returns one field
            for one type of table.
        """
        raise NotImplementedError

    def recordExists(self, id, table=TABLE_PAPERS):
        """
            Returns True if the specified record exists in the given table, False
            otherwise.
        """
        raise NotImplementedError

    def getMetadataByGUID(self, guid):
        """
            Returns a paper's metadata by GUID
        """
        raise NotImplementedError

    def getMetadataByField(self, field, value):
        """
            Returns a single paper's metadata by any field
        """
        raise NotImplementedError

    def listFieldByField(self, field1, field2, value):
        """
            Returns a list: for each paper, field1 if field2==value
        """
        raise NotImplementedError

    def updatePaper(self, metadata):
        """
            Updates an existing record in the db
        """
        raise NotImplementedError

    def listPapers(self, conditions=None):
        """
            Return a list of GUIDs in papers table where [conditions]. It's a
            SELECT query
        """
        raise NotImplementedError

    def runSingleValueQuery(self, query):
        raise NotImplementedError

    def addAuthor(self, author):
        """
            Make sure author is in database
        """
        raise NotImplementedError

    def addPaper(self, metadata, check_existing=True):
        """
            Adds a paper's metadata to the database

            :param metadata: paper's metadata
            :param check_existing: should we check that there's a match before adding it
        """
        raise NotImplementedError

    def addLink(self, GUID_from, GUID_to, authors_from, authors_to, year_from, year_to, numcitations):
        """
            Adds a link from a paper to another

        """
        raise NotImplementedError

    def addMissingPaper(self, metadata):
        """
            Inserts known data about an unkown paper

            :param metadata: metadata of missing paper
        """
        raise NotImplementedError

    def createDBindeces(self):
        """
            Call this after importing the metadata into the corpus and before
            matching in-collection references, it will speed up search massively
        """
        raise NotImplementedError

    def deleteAll(self, record_type):
        """
            WARNING! This function deletes all the records in a given "table" or
            of a given type.

            :param record_type: one of the tables that exist, e.g. [TABLE_PAPERS,TABLE_LINKS,TABLE_AUTHORS,TABLE_SCIDOCS,TABLE_CACHE]
        """
        raise NotImplementedError

    def deleteByQuery(self, record_type, query):
        """
            Delete the entries from a table that match the query.

            :param record_type: one of the tables that exist, e.g. [TABLE_PAPERS,TABLE_LINKS,TABLE_AUTHORS,TABLE_SCIDOCS,TABLE_CACHE]
            :type record_type: string
            :param query: a query to select documents to delete
            :type query: string
        """
        raise NotImplementedError

    def filterQuery(self, query):
        """
            Adds a global filter to the query so it only matches the selected
            collection, date, etc.

            :param query: string
        """
        return self.query_filter + " " + query

    def setCorpusFilter(self, collection_id=None, import_id=None, date=None):
        """
            Sets the filter query to limit all queries to a collection (corpus)
            or an import date

            :param collection: identifier of corpus, e.g. "ACL" or "PMC". This is set at import time.
            :type collection:basestring
            :param import: identifier of import, e.g. "initial"
            :type import:basestring
            :param date: comparison with a date, e.g. ">[date]", "<[date]",
            :type collection:basestring
        """
        raise NotImplementedError


def main():
    pass


if __name__ == '__main__':
    main()
