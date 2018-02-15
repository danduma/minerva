# Communicate with a ParsCit API server (see server.py) and convert
# its XML to metadata
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
from __future__ import absolute_import
from .read_parscit import ParsCitReader
import json
import requests


class ParsCitClient:
    """
        Connects to ParsCit web service, sends documents and converts the
        response to metadata.
    """
    def __init__(self, api_url):
        """
        """
        self.api_url = api_url
        if len(self.api_url) > 0 and self.api_url[-1] !="/":
            self.api_url += "/"
        self.reader=ParsCitReader()

    def tagFullDocument(self, doc_text):
        """
        """
        raise NotImplementedError


    def extractReferenceList(self, ref_list, format="raw"):
        """
            Main function: interface with ParsCit.

            Args:
                ref_list: list of strings to be processed. Each string can
                          contain multiple references
        """
        res=[]
        if isinstance(ref_list, list):
            ref_list ="\n\n".join(ref_list)

        # Hack so ParsCit will actually recognize the references
        if format=="raw":
            ref_list=u"References\n\n%s" % ref_list
        else:
            ref_list=u"<references>%s</references>" % ref_list

        data={"text":ref_list, "format":format}

        r=requests.post(self.api_url+"extract_citations/", json=data)

        if r.status_code != 200:
            # TODO specialized exceptions
            print(ref_list)
            raise ValueError("ParsCit exception")

        json_data=json.loads(r.content)
        res=self.reader.parseParsCitXML(json_data["parsed_xml"])
        return res

def simpleTest():
    from scidoc.reference_formatting import formatReference
    from parscit_server import test_data
    client=ParsCitClient("http://127.0.0.1:5000/parscit/")
    parsed=client.extractReferenceList([test_data])[0]
    for ref in parsed:
        print(formatReference(ref))
    pass


def main():
    simpleTest()
if __name__ == '__main__':
    main()
