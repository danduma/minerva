import pytextrank
import sys
import os

path_dir = "/Users/masterman/Downloads/pytextrank-master/"
path_stage0 = path_dir + "dat/mih.json"
path_stage1 = path_dir + "o1.json"

with open(path_stage1, 'w') as f:
    grafs = pytextrank.parse_doc(pytextrank.json_iter(path_stage0))
    for graf in grafs:
        f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
        # to view output in this notebook
        # print(pytextrank.pretty_print(graf))

# path_stage1 = path_dir + "o1.json"
path_stage2 = path_dir + "o2.json"

graph, ranks = pytextrank.text_rank(grafs)
pytextrank.render_ranks(graph, ranks)

with open(path_stage2, 'w') as f:
    for rl in pytextrank.normalize_key_phrases(grafs, ranks):
        f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
        # to view output in this notebook
        print(pytextrank.pretty_print(rl))

import networkx as nx
import pylab as plt

nx.draw(graph, with_labels=True)
plt.show()

path_stage1 = path_dir + "o1.json"
path_stage2 = path_dir + "o2.json"
path_stage3 = path_dir + "o3.json"

kernel = pytextrank.rank_kernel(path_stage2)

with open(path_stage3, 'w') as f:
    for s in pytextrank.top_sentences(kernel, path_stage1):
        f.write(pytextrank.pretty_print(s._asdict()))
        f.write("\n")
        # to view output in this notebook
        print(pytextrank.pretty_print(s._asdict()))

path_stage2 = path_dir + "o2.json"
path_stage3 = path_dir + "o3.json"

phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=12)]))
sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
s = []

for sent_text, idx in sent_iter:
    s.append(pytextrank.make_sentence(sent_text))

graf_text = " ".join(s)
print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))
