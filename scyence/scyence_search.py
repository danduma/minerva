import arxiv
import re
import json
import six

from proc.nlp_functions import cleanXML, tokenizeTextAndRemoveStopwords
from proc.query_extraction import WindowQueryExtractor

INSERT_CIT_TOKEN = "<$CIT>"

arxiv_cat = json.load(open("arxiv_cat.json"))

query_extractor = WindowQueryExtractor()
print("got this far")


def extract_query(input_text):
    query = input_text.replace(INSERT_CIT_TOKEN, "__cit")
    match = re.search(r"__cit", query, flags=re.DOTALL | re.IGNORECASE)

    params = {"parameters": [(20, 20)],
              "match_start": match.start(),
              "match_end": match.end(),
              "doctext": query,
              "extract_only_plain_text": True,
              }

    query = query_extractor.extractMulti(params)[0]["text"]
    print(query)
    # query = cleanXML(query)
    # query = tokenizeTextAndRemoveStopwords(query)
    return query


def clean_arxiv_string(text):
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


def format_author_string(authors):
    return ", ".join(authors)


def split_authors(authors):
    authors_struct = []
    for author in authors:
        names = author.split()
        if names[-1].lower() in ["ii", "iii"]:
            family = names[-2]
        else:
            family = names[-1]

        given = names[0]
        new_author = {"family": family, "given": given}
        authors_struct.append(new_author)
    return authors_struct


def formatAPA(authors, year):
    """
        Return only the authors for a citation formatted for APA-style bibliography
    """
    authors = split_authors(authors)

    res = u""
    if authors == []:
        res += u"?"
    elif len(authors) == 1:
        res += u'%s ' % authors[0]["family"]
    elif len(authors) == 2:
        res += u'%s and %s' % (authors[0]["family"], authors[1]["family"])
    else:
        try:
            res += authors[0]["family"] + u' et al.'
        except:
            res += six.text_type(authors[0]["family"], errors="replace")

    res += " (%s)" % year
    return res


def format_date_string(date):
    date = date.split("T")
    return date[0]

def get_year_from_date(date):
    bits = date.split("-")
    return bits[0]

def search_arxiv(input_text):
    search_query = extract_query(input_text)
    results = arxiv.query(search_query=search_query)
    res = []
    for result in results:
        year = get_year_from_date(result["published"])
        newres = {
            "id": result["id"],
            "title": clean_arxiv_string(result["title"]),
            "abstract": clean_arxiv_string(result["summary"]),
            "authors": format_author_string(result["authors"]),
            "dates": {
                "published": format_date_string(result["published"]),
                "updated": format_date_string(result["updated"]),
            },
            "date": format_date_string(result["published"]),
            "year": year,
            "affiliation": result["affiliation"],
            "doi": result["doi"],
            "category": result["arxiv_primary_category"]["term"],
            "apa_token": formatAPA(result["authors"], year)
        }
        if result["guidislink"]:
            newres["url"] = result["id"]
        elif "pdf_url" in result:
            newres["url"] = result["pdf_url"]
        elif "arxiv_url" in result:
            newres["url"] = result["arxiv_url"]

        res.append(newres)
    return res


def main():
    res = search_arxiv("quantum")


if __name__ == '__main__':
    main()
