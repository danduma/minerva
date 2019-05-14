from scyence.scyence_api import app
from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT

import db.corpora as cp

if __name__ == "__main__":
    import sys, getopt

    port = 5555

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:", ["dir=", "port="])
    except getopt.GetoptError:
        print('scyence_server.py -p <port>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--port"):
            port = int(arg)

    cp.Corpus.connectCorpus("", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    # getmetadata("a81ff731-0c6d-4b59-905d-fa8c1aab9c5e")

    print("Starting serving on port %s" % port)
    app.run(port=port, host='0.0.0.0', debug=False, use_reloader=True)
