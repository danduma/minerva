from models.keyword_features import FeaturesReader
from ml.document_features import en_nlp

import os
import spacy
from spacy import displacy
from spacy.tokens import Token

from bs4 import BeautifulSoup


def augment_features(context):
    for index, token in enumerate(context):
        token["index"] = index + 1
        token["abs_dist_cit"] = abs(token["dist_cit"])
    return context


def get_sentences_from_context(context):
    all_text = " ".join([x[0]["text"] for x in context])
    sentences1 = all_text.split("#STOP#")
    sentences = []
    for sentence in sentences1:
        sentence = sentence.strip()
        sentence = sentence.replace(" ,", ",").replace(" .", ".")
        if sentence == "":
            continue
        sentence = sentence[0].upper() + sentence[1:]
        sentences.append(sentence)
    return sentences


def get_sentence_tokens_from_token_list(tokens):
    cur_sentence = []
    sentences = []
    for token in tokens:
        if token["text"] == "#STOP#":
            sentences.append(cur_sentence)
            cur_sentence = []
        else:
            cur_sentence.append(token)

    return sentences


def get_sentence_text_from_tokens(tokens):
    all_text = " ".join([x["text"] for x in tokens])
    sentence = all_text.strip()
    sentence = sentence.replace(" ,", ",").replace(" .", ".")
    if sentence == "":
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


# def unroll_dependency_parse
# # https://stackoverflow.com/questions/32835291/how-to-find-the-shortest-dependency-path-between-two-words-in-python?rq=1
# import networkx as nx
# import spacy
# nlp = spacy.load('en')
#
# # https://spacy.io/docs/usage/processing-text
# document = nlp(u'Robots in popular culture are there to remind us of the awesomeness of unbound human agency.', parse=True)
#
# print('document: {0}'.format(document))
#
# # Load spacy's dependency tree into a networkx graph
# edges = []
# for token in document:
#     # FYI https://spacy.io/docs/api/token
#     for child in token.children:
#         edges.append(('{0}-{1}'.format(token.lower_,token.i),
#                       '{0}-{1}'.format(child.lower_,child.i)))
#
# graph = nx.Graph(edges)
#
# # https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html
# print(nx.shortest_path_length(graph, source='robots-0', target='awesomeness-11'))
# print(nx.shortest_path(graph, source='robots-0', target='awesomeness-11'))
# print(nx.shortest_path(graph, source='robots-0', target='agency-15'))

def get_spacy_parse(contexts):
    all_parses = []
    for context in contexts:
        tokens = context["tokens"]
        for index in range(len(tokens)):
            tokens[index]["extract"] = context["extract_mask"][index]
            tokens[index]["weight"] = context["weight_mask"][index]

        sentences = get_sentence_tokens_from_token_list(tokens)
        for sentence_tokens in sentences:
            sentence_text = get_sentence_text_from_tokens(sentence_tokens)
            parse = en_nlp(sentence_text)
            for index, token in enumerate(parse):
                # assuming same token index, as we'd already parsed it with the same spacy model before
                if token.text.strip().lower() != sentence_tokens[index]["text"].strip().lower():
                    print(token.text.strip().lower(), "!=", sentence_tokens[index]["text"].strip().lower())
                    print(parse[max(0, index - 7):min(len(parse) - 1, index + 7)])
                    print(" ".join(
                        [x["text"] for x in sentence_tokens[max(0, index - 7):min(len(parse) - 1, index + 7)]]))
                    break
                token._.extract = sentence_tokens[index]["extract"]
                token._.weight = sentence_tokens[index]["weight"]
                token._.dist_cit = sentence_tokens[index].get("dist_cit", 0)
                # token._.dist_cit_norm = sentence_tokens[index][0].get("dist_cit_norm", 0)

            display_dict = displacy.parse_deps(parse)

            # for index, word in enumerate(display_dict["words"]):
            #     if word["text"] != parse[index].text:
            #         print("ARGh")
            all_parses.append(parse)

    return all_parses


def render_all(all_parses):
    html = displacy.render(all_parses, style='dep', page=True, options={"collapse_punct": False})

    bs = BeautifulSoup(html, "lxml")
    all_figures = bs.findAll("figure")
    for findex, figure in enumerate(all_figures):
        all_words = figure.findAll("tspan", {"class": "displacy-word"})
        for windex, word in enumerate(all_words):
            if all_parses[findex][windex].text.strip() != word.text.strip():
                print("MAAC")
            if all_parses[findex][windex]._.extract:
                word["font-weight"] = "bold"
                word["font-size"] = "16pt"

    html = bs.prettify()
    with open("displacy_output.html", "w") as f:
        f.write(html)


def main():
    Token.set_extension("extract", default=False)
    Token.set_extension("weight", default=0.0)
    Token.set_extension("dist_cit", default=0)
    Token.set_extension("dist_cit_norm", default=0.0)

    exp_dir = "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace"
    features_data_filename = os.path.join(exp_dir, "feature_data.json.gz")

    contexts = FeaturesReader(features_data_filename, 10)

    render_all(get_spacy_parse(contexts))


if __name__ == '__main__':
    main()
