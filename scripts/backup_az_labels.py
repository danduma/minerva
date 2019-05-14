from db.ez_connect import ez_connect

from tqdm import tqdm


def backup_az_labels():
    corpus = ez_connect("AAC")
    guids = corpus.listPapers()


    for guid in tqdm(guids):
        doc = corpus.loadSciDoc(guid)
        for sent in doc.allsentences:
            sent["az_azprime"] = sent["az"]

        # for sent in doc.allsentences:
        #     assert sent["az_azprime"]

        corpus.saveSciDoc(doc)

def main():
    backup_az_labels()

if __name__ == '__main__':
    main()