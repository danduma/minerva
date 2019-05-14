# Scyence server
#
# Copyright:   (c) Daniel Duma 2019
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

"""
    AJAX frontend. Backend only sends info.
"""

from __future__ import print_function

from __future__ import absolute_import
import db.corpora as cp
from scyence.scyence_search import search_arxiv

from flask import Flask, request, Response, send_file, abort
import logging, os, json

import requests

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

##from flask.ext.compress import Compress
##Compress(app)

from multi.celery_app import set_config

ES_URL = set_config("aws-server")

from db.ez_connect import ez_connect


# corpus = ez_connect("AAC")

def return_send_file(filename="index.html"):
    full_path = os.path.join(os.getcwd(), "static", filename)
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        abort(404)


@app.route('/')
def get_root():
    """
        Returns the default file
    """
    return return_send_file()


@app.route('/<string:filename>')
def get_root_file(filename):
    """
        Serves files on the root dir
    """
    if filename == "":
        filename = "template.html"

    return return_send_file(filename)


@app.route('/<path:path>')
def get_path(path):
    """
        Serves files not on the root dir
    """
    return return_send_file(path)


@app.route('/ccr', methods=['POST'])
def ccr():
    """
        Returns recommendations based on draft text. The insertion token is marked by <insert_citation>
    """

    logging.info("Finding relevant papers for %s " % request)
    data = request.json
    print(data)

    if "collection" not in data:
        collection = "arxiv"
    else:
        collection = data["collection"].lower()
        if collection not in ["arxiv"]:
            error = "Unknown collection id" + data["collection"]
            print(error)
            return json.dumps({"error": error})

    if "text" not in data:
        error = "text parameter is required"
        print(error)
        return json.dumps({"error": error})

    if collection == "arxiv":
        results = search_arxiv(data["text"])

    res = {"results": results}

    return Response(json.dumps(res), mimetype='application/json')


@app.route('/getmetadata/<string:paper_id>')
def getmetadata(paper_id):
    """
        Returns paper metadadata
    """
    logging.info("Loading metadata for %s " % paper_id)
    meta = cp.Corpus.getMetadataByGUID(paper_id)
    if not meta:
        return json.dumps({"error": "GUID not found", "error_code": 404})

    del meta["inlinks"]
    del meta["outlinks"]
    return Response(json.dumps({"metadata": meta}), mimetype='application/json')


@app.route('/bulkmetadata/<string:papers>')
def bulkmetadata(papers):
    """
        Returns bulk metadata given a list of guids
    """
    guids = [p for p in papers.split(",") if p != ""]
    logging.info("Getting bulk metadata for %d guids " % len(guids))
    res = []
    for guid in guids:
        meta = cp.Corpus.getMetadataByGUID(guid)
        if not meta:
            res.append({"error": "GUID not found", "error_code": 404})

        del meta["inlinks"]
        del meta["outlinks"]
        res.append({"metadata": meta})

    return Response(json.dumps(res), mimetype='application/json')


@app.route('/getbows/<string:paper_id>')
def getbows(paper_id):
    """
        Returns available BOWs for paper
    """
    logging.info("Loading bows for %s " % paper_id)

    bows = cp.Corpus.SQLQuery("select ID from cache where ID like \"bow_{}_%\" ".format(paper_id))

    return json.dumps({"bows": bows})


@app.route('/graph/<string:paper_id>')
def getgraph(paper_id):
    """
        Returns nodes in the graph. As a tree?
    """
    logging.info("Querying nodes for %s " % paper_id)
    meta = cp.Corpus.getMetadataByGUID(paper_id)
    if not meta:
        return json.dumps({"error": "GUID not found", "error_code": 404})

    ##    bows=cp.Corpus.SQLQuery("select ID from cache where ID like \"bow_{}_%\" ".format(meta["guid"]))
    return json.dumps({"metadata": meta})


@app.route('/es_query/<path:index>', methods=["POST"])
def es_query(index):
    """
        This is just a redirect to the Elasticsearch server. This is to bypass the same-origin policy nonsense.
    """
    logging.info("es_query received: %s" % request.data)
    logging.info("es_query for index: %s" % index)
    r = requests.post("http://%s/%s" % (ES_URL, index), data=request.data)
    return Response(r.text, mimetype='application/json')
