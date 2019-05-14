from db.elastic_corpus import ElasticCorpus
from tqdm import tqdm
from multi.config import set_config

# koko
# server1 = "http://129.215.197.75:9200/"

# es2
# server2 = "http://elastic:bvSHXy4k@35.197.226.158:9200/"

# es5
# server2 = "http://elastic:bvSHXy4k@35.234.156.47:9200/"

# server1 = "http://elastic:bvSHXy4k@35.234.156.47:9200/" # es5
# server2 = "http://elastic:bvSHXy4k@35.197.226.158:9200/" # es2

# server1 = "http://elastic:bvSHXy4k@10.154.0.5:9200/" # es5
# server2 = "http://10.154.0.7:8080/" # es2

server1 = set_config("koko")
# server2=set_config("koko")

c1 = ElasticCorpus()
c1.default_timeout = 360
c1.connectCorpus("", server1)

# c2 = ElasticCorpus()
# c2.connectCorpus("", server2)

print("Getting list of papers")


# all_files = c1.listPapers(max_results=100000000)


# def mega_query():
#     res = self.es.search(
#         *args,
#         size=size,
#         sort=sort,
#         scroll=scroll_time,
#         **kwargs
#     )
#
#     results = res['hits']['hits']
#     scroll_size = res['hits']['total']
#     while (scroll_size > 0) and len(results) < self.max_results:
#         try:
#             scroll_id = res['_scroll_id']
#             rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
#             res = rs
#             results.extend(rs['hits']['hits'])
#             scroll_size = len(rs['hits']['hits'])
#         except Exception as e:
#             print(e)
#             break

def update_scidocs():
    # manually update /scidocs
    all_files = c1.listPapers(max_results=100000000)
    for guid in tqdm(all_files):
        if not c2.es.exists(index="scidocs", doc_type="scidoc", id=guid):
            print(guid, "missing. Uploading")
            doc = c1.loadSciDoc(guid)
            c2.saveSciDoc(doc)


def update_papers():
    # manually update /papers
    all_files = c1.listPapers(max_results=100000000)
    for guid in tqdm(all_files):
        meta = c1.getMetadataByGUID(guid)
        c2.addPaper(meta, check_existing=False)


def update_cache_yield(index="cache"):
    # manually update /cache
    total_files = None

    for batch, scroll_size in c1.yieldingUnlimitedQuery(body={"query": {"match_all": {}}}, _source=False,
                                                        index=index):
        if total_files is None:
            total_files = scroll_size
            pbar = tqdm(range(scroll_size), desc="Uploading cached BOWs")

        for path in batch:
            path = path["_id"]
            if not c2.es.exists(index="cache", doc_type="cache", id=path):
                print(path, "does not exist. Uploading.")
                data = c1.loadCachedJson(path)
                c2.saveCachedJson(path, data)

            pbar.update()


def update_cache_bulk():
    # manually update /cache
    total_files = None

    print("Getting list from server 1...")
    all_cached_files = [x["_id"] for x in
                        c1.unlimitedQuery(body={"query": {"match_all": {}}}, _source=False, index="cache")]
    print("Getting list from server 2...")
    all_uploaded_files = [x["_id"] for x in
                          c2.unlimitedQuery(body={"query": {"match_all": {}}}, _source=False, index="cache")]

    all_cached_files = set(all_cached_files)
    all_uploaded_files = set(all_uploaded_files)

    new_files = all_cached_files - all_uploaded_files

    print("Uploading new files")
    for path in tqdm(new_files, "Uploading"):
        data = c1.loadCachedJson(path)
        c2.saveCachedJson(path, data)


def check_broken_scidocs():
    all_files = c2.listPapers(max_results=10)
    broken_files = 0

    # all_files=["e2138958-7a39-4aa9-ace0-5e7f258c9d0a", "125212a9-6dc8-436a-8426-a46a54ff32cd"]
    for guid in tqdm(all_files):
        try:
            doc = c2.loadSciDoc(guid)
            if doc is None:
                raise ValueError("Doc is None")
        except ValueError as e:
            print(e)
            print(guid, "broken. Uploading")
            doc = c1.loadSciDoc(guid)
            c2.saveSciDoc(doc)
            broken_files += 1

    print("Broken files", broken_files)


def copy_papers():
    # manually update /cache
    total_files = None

    print("Getting list of papers...")
    all_guids = c1.listPapers()
    #
    # for guid in all_guids:
    #     res = cp.Corpus.es.get(
    #         index="papers",
    #         doc_type="paper",
    #         id=guid,
    #     )
    #
    #     cp.Corpus.es.index(index="papers2")

    print("Copying files")
    for guid in tqdm(all_guids, "Copying"):
        paper = c1.es.get(
            index="papers",
            doc_type="paper",
            id=guid,
        )

        c1.es.index(
            index="papers2",
            doc_type="paper",
            id=guid,
            op_type="index",
            body=paper["_source"],
        )


copy_papers()
