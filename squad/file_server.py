# Very simple file server
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from flask import Flask, request, send_from_directory, send_file
from flask.ext.compress import Compress
import logging, os

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
##app.debug=True
logging.basicConfig(level=logging.INFO)

Compress(app)

BASE_SERVING_DIR=r"g:\nlp\phd\pmc_coresc\inputXML"

@app.route('/file/<path:path>')
def file_route(path):
    if path.startswith(BASE_SERVING_DIR):
        full_path=path
    else:
        full_path=os.path.join(BASE_SERVING_DIR,path)

    logging.info("Sending %s to %s" % (full_path, request.remote_addr))
##    return send_from_directory(BASE_SERVING_DIR, path)
    return send_file(full_path)

if __name__ == "__main__":
    import sys, getopt

    port=5599

    try:
        opts, args = getopt.getopt(sys.argv[1:],"d:p:",["dir=","port="])
    except getopt.GetoptError:
        print('file_server.py -d <serving_directory> -p <port>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-d", "--dir"):
            BASE_SERVING_DIR = arg
        elif opt in ("-p", "--port"):
            port = arg

    print("Starting serving files from %s on port %s" % (BASE_SERVING_DIR,port))
    app.run(port=port)
