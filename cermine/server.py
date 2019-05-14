# A simple API server for ParsCit using Flask
#
# Copyright:  (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
from flask import Flask, jsonify, Blueprint, request, Response, url_for
import logging, subprocess, os, tempfile, json, sys, codecs

import unicodedata

parscit_bp = Blueprint("parscit", __name__, url_prefix="/parscit")

PATH_TO_PARSCIT_BIN=r"G:\NLP\ParsCit-win\bin"
PATH_TO_PARSCIT=PATH_TO_PARSCIT_BIN+os.sep+"citeExtract.pl"

def extract_core(text, mode, format="raw"):
    """
        Function that calls citeExtract.pl

        Inspired by ParsCitServer.rb included with ParsCit

        Args:
            text: the full text to pass to parscit
            mode: "extract_citations", "extract_header", "extract_section", "extract_meta", or "extract_all"
            format: "raw" or "xml"
    """

    # Tried to send the text via stdin, but the perl script would have none of it
    # according to the documentation this is accepted but it doesn't work
    tmp_file = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
    tmp_file.close()

    tmp_file=codecs.open(tmp_file.name,"w",encoding="utf-8",errors="ignore")
    tmp_file.write(text)
    tmp_file.close()

    file_name=tmp_file.name
    print(file_name)
    pipe = subprocess.Popen(
        ["perl",    # so it runs on Windows
        PATH_TO_PARSCIT,
        "-q",   # quiet mode, output just xml
        "-m",  mode,
        "-i" , format,
        file_name
        ],

    stdout=subprocess.PIPE
    )

    stdout,stderr=pipe.communicate()
    result = stdout
    tmp_file.close()
    try:
        filename, file_extension = os.path.splitext(tmp_file.name)
        os.remove(tmp_file.name)
        os.remove(filename+".cite")
        os.remove(filename+".body")
    except:
        print("Warning: couldn't remove some temp files")

    return result


@parscit_bp.route("/<string:mode>/", methods=["POST"])
def api_entrypoint(mode):
    """
        Main (and only) API entry point.

        Args:
            mode: see below
    """
    if mode not in ["extract_citations", "extract_header", "extract_section", "extract_meta", "extract_all"]:
        resp=jsonify({})
        resp.status_code=405 # method not allowed
        return resp

    print(request)

    try:
        data=json.loads(request.data)
    except:
        resp=jsonify(error="Wrong JSON format")
        resp.status_code=400 # bad request
        return resp

    format=data.get('format', 'raw')
    text = data.get('text', u'')

    text = unicodedata.normalize('NFKD', text)

    parsed=extract_core(text,mode,format)

    resp=jsonify(parsed_xml=parsed)
    resp.status_code=200 # all good
    return resp

def startStandaloneServer(host="localhost", port=5000):
    """
        Starts a basic server with the blueprint
    """
    app = Flask(__name__)
    app.register_blueprint(parscit_bp, url_prefix="/parscit")

    print("Running ParsCit API on %s:%d" % (host,port))
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
##    app.logger.setLevel(logging.)
    app.run(host=host, port=port, debug=False, threaded=True)


test_data=u"""References

Allen, J. & Perrault, C. (1980). Analyzing intention in utterances. Artificial Intelligence,15, 143-178.

Austin, J. (1962). How to do things with words. Cambridge, MA: Harvard University Press.Cohen, P. & Perrault, C. (1979).

Elements of a plan-based theory of speech acts. CognitiveScience, 3, 177-212.

DeJong, G. & Mooney, R. (1986). Explanation based learning: An alternate view. Machine Learning, 1, 145-176.

Fikes, R. & Nilsson, N. (1971). STRIPS: A new approach to the application of theorem proving to problem solving. Artificial Intelligence, 2, 189-208.

Gibbs, R. (1983). Do people always process the literal meaning of indirect requests? Journal of Experimental Psychology: Learning, Memory, and Cognition, 3, 524-533.

Gibbs, R. (1984). Literal meaning and psychological theory. Cognitive Science, 8, 275-305.

Grosz, B. & Sidner, C. (1986). Attention, intentions and the structure of discourse. American Journal of Computational Linguistics, 12, 175-204.

Hinkleman, E. & Allen, J. (1988). How to do things with words, computationally speaking."""

def simpleTest():
    # Testing
    print(extract_core(test_data2,"extract_citations"))

def main():
    startStandaloneServer()
    pass

if __name__ == '__main__':
    main()
