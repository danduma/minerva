from models.keyword_features import FeaturesReader
# from ml.document_features import en_nlp

import os

# import spacy
# from spacy import displacy
# from spacy.tokens import Token

from colour import Color


def getRank(mrr_score):
    if mrr_score == 0:
        return -1
    return 1 / mrr_score


def getColourScale(weight, max):
    col1 = Color("purple")
    if max == 0:
        return "", "#FFF"

    val = 1 - ((weight / max) * 0.6)
    text_col = ""

    if val < 0.2:
        text_col = "#FFF"

    col1.set_luminance(val)
    return text_col, col1.get_hex()


def render_all(contexts):
    res = []
    for context in contexts:
        all_text = []
        context["weight_mask"] = [k for k in context["weight_mask"] if k is not None]
        try:
            max_weight = max(context["weight_mask"])
        except:
            print("arg")

        for index, token in enumerate(context["tokens"]):
            token_text = token["text"]

            # if context["extract_mask"][index]:
            #     token_text = "<b>%s</b>" % token_text

            if token["token_type"] == "c":
                token_text = "<span class=\"citation\">CITATION (%d)</span>" % context["citation_multi"]
            elif token["token_type"] == "t" and context["extract_mask"][index]:
                text_col, back_col = getColourScale(context["weight_mask"][index], max_weight)

                if text_col != "":
                    text_col = "color:" + text_col + ";"

                token_text = "<span style=\"background-color:%s;%s\">%s</span>" % (
                    back_col, text_col, token_text)

            if token["text"] == "__author":
                token_text = "<span class=\"author\">AUTHOR</span>"

            all_text.append(token_text)

        para_text = " ".join(all_text)
        sentences1 = para_text.split("#STOP#")
        sentences = []
        for sentence in sentences1:
            sentence = sentence.strip()
            sentence = sentence.replace(" ,", ",").replace(" .", ".")
            if sentence == "":
                continue
            sentence = sentence[0].upper() + sentence[1:]
            sentences.append(sentence)

        para_text = " ".join(sentences)

        side_bar = """<table><tr><td><b>Rank</b></td></tr> 
                <tr><td>Baseline:<td>%d</td></tr> 
                <tr><td>Boost=1:<td>%d</td></tr> 
                <tr><td>Adj. boost:<td>%d</td></tr> 
                </table>
                   """ % (
            getRank(context["original_scores"]["mrr_score"]),
            getRank(context["kw_selection_scores"]["ndcg_score"]),
            getRank(context["kw_selection_weight_scores"]["ndcg_score"]),
        )

        selectors_title = {"MultiMaximalSetSelector": "Maximal keyword selector",
                           "MinimalSetSelector": "Minimal keyword selector",
                           "AllSelector": "Keyword weighting",
                           }
        context_title = ""
        if "keyword_selection_entry" in context:
            context_title = selectors_title.get(context["keyword_selection_entry"], context["keyword_selection_entry"])

        context_html = """<div class=\"context\">
        %s
        <table><tr>
        <td width=\"85%%\"><p>%s</p></td>
        <td><p>%s</p></td>
        </tr></table></div>""" % (context_title, para_text, side_bar)
        res.append(context_html)

    html = u"<html><head><meta charset=\"utf-8\" /> " \
           u"<link href=\"context_vis.css\" rel='stylesheet'>" \
           u"</head><body>%s</body></html>" % " ".join(res)
    return html


def generateHTMLVisForFeatures(features_data_filename, max_items=100):
    exp_dir = os.path.dirname(features_data_filename)
    contexts = FeaturesReader(features_data_filename, max_items)
    html = render_all(contexts.getIterator())

    html_filename = os.path.splitext(os.path.basename(features_data_filename))[0] + ".html"
    with open(os.path.join(exp_dir, html_filename), "w") as f:
        f.write(html)


def main():
    # Token.set_extension("extract", default=False)
    # Token.set_extension("weight", default=0.0)
    # Token.set_extension("dist_cit", default=0)
    # Token.set_extension("dist_cit_norm", default=0.0)

    exp_dir = "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace"
    # exp_dir = "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace"

    # features_data_filename = os.path.join(exp_dir, "feature_data_w.json.gz")
    # features_data_filename = os.path.join(exp_dir, "feature_data_test_w.json.gz.fixed")
    # features_data_filename = os.path.join(exp_dir, "feature_data_at_w_min1.json.gz")
    features_data_filename = os.path.join(exp_dir, "thesis_example_selection.json")
    # features_data_filename = os.path.join(exp_dir, "feature_data_test_at_w_1u1d.json.gz")

    generateHTMLVisForFeatures(features_data_filename, 100)


if __name__ == '__main__':
    main()
