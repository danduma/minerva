# Corpus class. Deals with a database of scientific papers.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import os, sys, re, json, glob, codecs, uuid, unicodedata
from minerva.proc.general_utils import (AttributeDict, ensureTrailingBackslash,
ensureDirExists, normalizeTitle, removeSymbols)
from minerva.scidoc.scidoc import SciDoc

class BaseReferenceMatcher(object):
    """
        Base class for ReferenceMatcher. This class provides the matchReference
        function, which must find the right reference in the corpus
    """
    def __init__(self, corpus):
        """
        """
        self.corpus=corpus

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
        if author.get("author_id","") not in ["",None]:
            author_record=self.corpus.getAuthorByField("author_id",author["author_id"])
            if author_record:
                return author_record

        for field in ["given","middle","family"]:
            if field in author:
                author[field]=removeSymbols(author[field])

        query="SELECT author FROM authors WHERE author.given=\"%s\" and author.family=\"%s\""% (author["given"],author["family"])
        if author.get("middle","") != "":
            query+=" and author.middle=\"%s\"" % author["middle"]

        matches=self.corpus.SQLQuery(query)

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

            Args:
                ref: reference dict
                doc: SciDoc (optional). Only here to enable decendant
        """
        self.corpus.checkConnectedToDB()

        # try matching by ID first
        for id_type in [("doi","doi"),("pmid","corpus_id"),("corpus_id","corpus_id"),("guid","guid")]:
            if ref.get(id_type[0],"") not in ["",None]:
                doc_meta=self.corpus.getMetadataByField(id_type[1],ref[id_type[0]])
                if doc_meta:
                    return doc_meta

        # if it can't be matched by id
        norm_title=normalizeTitle(ref["title"])

        if not isinstance(norm_title, unicode):
            norm_title=unicode(norm_title, errors="ignore")

        rows=self.corpus.listFieldByField("metadata","norm_title",norm_title)

        for row in rows:
            doc_meta=row
##            doc_meta=json.loads(row[0]) # load metadata dict
            if len(doc_meta["surnames"]) > 0:
                for a1 in doc_meta["surnames"]:
                    for a2 in ref["surnames"]:
                        if a1 and a2 and a1.lower() == a2.lower():
                            # essentially, if ANY surname matches
                            return doc_meta
        return None


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
        self.annotators={}
        # pretty-print saved SciDoc files?
        self.saveDocIndent=None
        self.setPaths("")
        self.matcher=DefaultReferenceMatcher(self)
        # any extra data to hold
        self.global_counters={}
        self.AUTO_ADD_AUTHORS=True

    def setPaths(self, root_dir):
        """
            Basic setPaths, mostly here for cachedJson
        """
        self.ROOT_DIR=root_dir
        self.paths=AttributeDict()
        self.paths.inputXML=self.ROOT_DIR+"inputXML"+os.sep
        self.paths.fileDB=self.ROOT_DIR+"fileDB"+os.sep
        self.paths.inputXML=self.ROOT_DIR+"inputXML"+os.sep
        self.paths.experiments=self.ROOT_DIR+"experiments"+os.sep
        self.paths.output=self.ROOT_DIR+"output"+os.sep

    def connectCorpus(self, base_directory, initializing_corpus=False, suppress_error=False):
        """
            If DB has been created, connect to it. If not, initialize it first.

            Args:
                base_directory: root dir of this corpus
                initializing_corpus: if True, create DB and directories
                suppress_error: if true, db doesn't complain if it's connected already
        """
        raise NotImplementedError

    def loadAnnotators(self,annotators=["AZ","CFC"]):
        """
            Loads annotators (classifiers), either as parameter or by default
        """
        for annotator in annotators:
            if annotator=="AZ":
                from minerva.az.az_cfc_classification import AZannotator
                self.annotators[annotator]=AZannotator("trained_az_classifier.pickle")
            if annotator=="CFC":
                from minerva.az.az_cfc_classification import CFCannotator
                self.annotators[annotator]=CFCannotator("trained_cfc_classifier.pickle")

    def annotateDoc(self,doc,annotate_with=None):
        """
            Shortcut to get Corpus object to annotate file with one or more annotators
        """
        if not annotate_with:
            annotate_with=self.annotators.keys()

        for annotator in annotate_with:
            assert(self.annotators.has_key(annotator))
            self.annotators[annotator].annotateDoc(doc)

    def generateGUID(self, metadata=None):
        """
            Generates a UUID version 4 (random).
        """
        hex_string=str(uuid.uuid4())
        return hex_string

    def generateAuthorID(self, author=None):
        """
            Generates a UUID version 4 (random).
        """
        hex_string=str(uuid.uuid4())
        return hex_string

    def generateDeterministicGUID(self, metadata):
        """
            Generates a UUID version 5. It is based on the authors' surnames and
            title of the document

            >>> print LocalCorpus().generateGUID({"surnames":[u"Jones",u"Smith"], "year": 2005, "norm_title": u"a simple title"})
            296da00b-5fb9-5800-8b5f-f50ef951afc4
        """
        if metadata.get("norm_title","") == "" or metadata.get("surnames",[]) == []:
            raise ValueError("No norm_title or no surnames!")

        uuid_string = " ".join(metadata["surnames"]).lower().strip()
        uuid_string += metadata["norm_title"]
        uuid_string = unicodedata.normalize("NFKC",uuid_string).encode("utf-8")
        hex_string=str(uuid.uuid5(uuid.NAMESPACE_DNS,uuid_string))
        return hex_string


    def updateAllPapersMetadataOutlinks(self):
        """
            Goes through the papers table, updating the outlinks in the metadata
            field with the actual in-collection references.

            Returns a list of guids of documents that have experienced changes,
            so that indeces can be rebuilt accordingly if neccesary
        """

        res=[]
        for paper_guid in self.listPapers():
            metadata=self.getMetadataByGUID(paper_guid)
            old_outlinks=metadata["outlinks"]
            doc=self.loadSciDoc(metadata["guid"])
            new_outlinks=self.listDocInCollectionReferences(doc)

            if set(new_outlinks) != set(old_outlinks):
                res.append(paper_guid)
            metadata["outlinks"]=new_outlinks
            self.updatePaper(metadata)

        return res

    def listDocInCollectionReferences(self,doc):
        """
            Returns a list of guids of all the in-collection references of a doc
        """
        res=[]
        for ref in doc["references"]:
            match=self.matcher.matchReference(ref)
            if match:
                res.append(match["guid"])
        return res

    def tagAllReferencesAsInCollectionOrNot(self, doc):
        """
            For each reference in a doc, it adds a boolean "in_collection"
        """
        for ref in doc["references"]:
            match=self.matcher.matchReference(ref)
            ref["in_collection"]=True if match else False


    def listIncollectionReferencesOfList(self, guid_list):
        """
            Will return all in-collection references of all the files in the list
        """
        res=[]
        assert isinstance(guid_list, list)
        for guid in guid_list:
            res.extend(self.getMetadataByGUID(guid)["outlinks"])
        res=list(set(res))
        return res

    def matchAllReferences(self, doc):
        """
            Matches each reference in a SciDoc with its guid in the collection,
            if found.

            :type doc:SciDoc
        """
        for ref in doc.references:
            ref["guid"]=None
            match=self.matcher.matchReference(ref, doc)
            if match:
                ref["guid"]=match["guid"]


    def selectDocResolvableCitations(self, doc, year=None):
        """
            Returns a list of {"cit","match"} with the citations in the text and
            their matching document in the index

            WARNING! Also removes inline citations that aren't resolvable

            :param doc: SciDoc
            :type doc: SciDoc
            :returns: tuple of (resolvable_citations, outlinks, missing_references)
            :param year: Maximum year to select from
        """
        res=[]
        unique={}
        sents_with_multicitations=[]
        missing_references=[]
        if not year:
            year=9999

        for ref in doc["references"]:
            match=self.matcher.matchReference(ref, doc)
            if match and int(match["year"]) <= year:
                ref["guid"]=match["guid"]
                for cit_id in ref.get("citations",[]):
                    res.append({"cit":doc.citation_by_id[cit_id],"match_guid":match["guid"]})
                    unique[match["guid"]]=unique.get(match["guid"],0)+1
            else:
                missing_references.append(ref)
                #remove citation from sentence
                for cit_id in ref.get("citations",[]):
                    cit=doc.citation_by_id[cit_id]
                    sent=doc.element_by_id[cit["parent_s"]]
                    sent["text"]=re.sub(r"<cit\sid=.?"+str(cit["id"])+".{0,3}/>","",sent["text"], 0,re.IGNORECASE)
                    sents_with_multicitations.append(sent)

        for sent in sents_with_multicitations:
            # deal with many citations within characters of each other: make them know they are a cluster
            doc.countMultiCitations(sent)

        return [res, unique, missing_references]

    def selectBOWParametersToPrebuild(self, guid, method, parameters):
        """
            Returns a list of parameters to build BOWs for, which may be empty,
            by removing all those that already have a prebuiltBOW

            Delete the files if you want them prebuilt again
        """
        newparam=[param for param in parameters if not
            self.cachedJsonExists("bow", guid,
                {"method":method, "parameter": param}
                )]
        return newparam

    def savePrebuiltBOW(self, guid, params, bow):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path=self.cachedDataIDString("bow", guid, params)
        self.saveCachedJson(path, bow)

    def loadPrebuiltBOW(self, guid, params):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path=self.cachedDataIDString("bow", guid, params)
        return self.loadCachedJson(path)

    def saveResolvableCitations(self, guid, citations_data):
        """
            Wrapper function to make the code cleaner
        """
        guid=guid.lower()
        self.saveCachedJson(self.cachedDataIDString("resolvable",guid),citations_data)

    def loadResolvableCitations(self,guid):
        """
            Wrapper function to make the code cleaner
        """
        guid=guid.lower()
        return self.loadCachedJson(self.cachedDataIDString("resolvable",guid))

    def loadOrGenerateResolvableCitations(self, doc_file, year=None):
        """
            Tries to open pre-computed resolvable citations and outlinks,
            generates and saves them if not found

            Args:
                doc_file: SciDoc
        """
        missing_references=[]
        if not self.cachedJsonExists("resolvable",doc_file["metadata"]["guid"]):
            resolvable, outlinks, missing_references=self.selectDocResolvableCitations(doc_file, year)
            citations_data={"resolvable":resolvable,"outlinks":outlinks,"missing_references":missing_references}
            self.saveResolvableCitations(doc_file["metadata"]["guid"],citations_data)
        else:
            citations_data=self.loadResolvableCitations(doc_file["metadata"]["guid"])
            if not citations_data:
                resolvable, outlinks, missing_references=self.selectDocResolvableCitations(doc_file)

        return citations_data

    def checkConnectedToDB(self):
        """
            Asserts that the DB is connected
        """
        assert self.connectedToDB(), "Not connected to DB!"


#-------------------------------------------------------------------------------
#   Subclass-specific functions from here on
#-------------------------------------------------------------------------------
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
        if type=="resolvable":
            return type+"_"+guid
        if type=="bow":
            return type+"_"+guid+"_"+params["method"]+unicode(params["parameter"])

    def saveCachedJson(self, path, data):
        """
            Save anything as JSON
        """
        raise NotImplementedError

    def loadCachedJson(self,path):
        """
            Load precomputed JSON
        """
        raise NotImplementedError

    def loadSciDoc(self,guid):
        """
            If a SciDocJSON file exists for guid, it returns it, otherwise None
        """
        raise NotImplementedError

    def saveSciDoc(self,doc):
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

    def getRecord(self, id, table="papers"):
        """
            Abstracts over getting data from a row in the db. Returns all the
            fields of the record for one type of table.

            :param id: id of the record
            :param table: table alias, e.g. ["papers", "scidocs"]
        """
        raise NotImplementedError

    def getRecordField(self, id, table="papers"):
        """
            Abstracts over getting data from a row in the db. Returns one field
            for one type of table.
        """
        raise NotImplementedError

    def recordExists(self, id, table="papers"):
        """
            Returns True if the specified record exists in the given table, False
            otherwise.
        """
        raise NotImplementedError

    def getMetadataByGUID(self,guid):
        """
            Returns a paper's metadata by GUID
        """
        raise NotImplementedError

    def getMetadataByField(self,field,value):
        """
            Returns a single paper's metadata by any field
        """
        raise NotImplementedError

    def listFieldByField(self,field1,field2,value):
        """
            Returns a list: for each paper, field1 if field2==value
        """
        raise NotImplementedError


    def updatePaper(self,metadata):
        """
            Updates an existing record in the db
        """
        raise NotImplementedError

    def listPapers(self,conditions):
        """
            Return a list of GUIDs in papers table where [conditions]. It's a
            SELECT query
        """
        raise NotImplementedError

    def runSingleValueQuery(self,query):
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

    def addLink(self,GUID_from,GUID_to,authors_from,authors_to,year_from,year_to,numcitations):
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

            :param record_type: one of the tables that exist, e.g. ["papers","links","authors","scidocs","cached"]
        """
        raise NotImplementedError

    def deleteByQuery(self, record_type, query):
        """
            Delete the entries from a table that match the query.

            :param record_type: one of the tables that exist, e.g. ["papers","links","authors","scidocs","cached"]
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
        return self.query_filter+" "+query

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
