import json

import requests
from elasticsearch import ConnectionError
import six

def getElasticURL(endpoint, index_name, doc_type):
    """
        Builds the url string for a given index_name and doc_type using the
        endpoint, and deals with it being a string and a dict
    """
    if isinstance(endpoint, six.string_types):
        url = endpoint + index_name + "/" + doc_type + "/"
    else:
        url = "http://%s:%d/%s/%s/" % (endpoint["host"],
                                       endpoint["port"],
                                       index_name,
                                       doc_type
                                       )
    return url


def getElasticTotalDocs(endpoint, index_name, doc_type):
    """
        Returns total number of documents in an index
    """
    url = getElasticURL(endpoint, index_name, doc_type)
    r = requests.get(url + "_count", {"query": {"match_all": {}}})
    data = json.loads(r.text)
    docs_in_index = int(data["count"])
    return docs_in_index


def getElasticTermScores(endpoint, token_string, index_name, field_names, doc_type="doc"):
    """
        Returns doc_freq and ttf (total term frequency) from the elastic index
        for each field for each token

        :param token_string: a string containing all the tokens
        :index_name: name of the index we're querying
        :field_names: a list of fields to query for
        :returns: a dict[field][token]{
    """
    url = getElasticURL(endpoint, index_name, doc_type) + "_termvectors"

    if isinstance(field_names, six.string_types):
        field_names = [field_names]

    doc = {field_name: token_string[:200] for field_name in field_names}
    request = {
        "term_statistics": True,
        "field_statistics": False,
        "positions": False,
        "offsets": False,
        "payloads": False,
        "dfs": True,
        "doc": doc
    }
    r = requests.post(url, data=json.dumps(request))
    data = json.loads(r.text)

    res = {}
    if "error" in data:
        print(data)
        raise ConnectionError(data["error"]["reason"])

    if data["found"]:
        for field in data["term_vectors"]:
            res[field] = {}
            for token in data["term_vectors"][field]:
                res[field][token] = data["term_vectors"][field][token]
    ##                del res[field][token]["term_freq"]

    ##    assert False
    return res