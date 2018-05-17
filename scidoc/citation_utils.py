# Functions for automatically labelling, extracting and matching citations
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from __future__ import absolute_import
import re
from proc.general_utils import levenshtein
import six
from six.moves import range

jatsmultinumbercitation = re.compile(
    r"\[.*<xref\s+?ref-type=\"bibr\".*?>(.+?)</xref>\s*?(-|--|&#x02013;)<xref\s+?ref-type=\"bibr\".*?>(.+?)</xref>",
    re.IGNORECASE)

rxauthors = re.compile(r"(<REFERENCE>.*?\n)(.*?)(<DATE>)", re.IGNORECASE | re.DOTALL)
rxtitle = re.compile(r"(</DATE>.*?\n)(.*?)(\.|</REFERENCE>)", re.IGNORECASE | re.DOTALL)
rxdate = re.compile(r"(<DATE>)(.*?)(</DATE>)", re.IGNORECASE | re.DOTALL)

rxsingleauthor = re.compile(r"(<SURNAME>)(.*?)(</SURNAME>)", re.IGNORECASE | re.DOTALL)
##rxsingleyear=re.compile(r"\d{4}\w{0,1}", re.IGNORECASE | re.DOTALL)
rxsingleyear = re.compile(r"in\spress|to\sappear|forthcoming|submitted|\d{4}\w{0,1}", re.IGNORECASE | re.DOTALL)
rxwtwoauthors = re.compile(r"(\w+)\sand\s(\w+)", re.IGNORECASE | re.DOTALL)
rxetal = re.compile(r"(\w+)\set\sal", re.IGNORECASE | re.DOTALL)

rxseparatebrackets = re.compile(r"(?:(?:\[.*?(\d+)\][\s,]+\[(\d+)[\-\,\s]?))", re.IGNORECASE)

apa_pre_family_name = "de |von |van "
apa_author = "((?:" + apa_pre_family_name + ")?[A-Z][A-Za-z'`-]+)"
apa_etal = "(et al.?)"
##apa_additional = "(?:,? (?:(?:and |& )?" + apa_author + "|" + apa_etal + "))"
apa_additional = "(?:,? *" + apa_author + "? (?:(?:and |& |&amp; )?" + apa_author + "|" + apa_etal + "))"
apa_year_num = "((?:19|20)[0-9][0-9][a-g]?|in\spress|to\sappear|forthcoming|submitted)"
apa_page_num = "(?:, p.? [0-9]+)?"  # Always optional
apa_year = "(?:,? *" + apa_year_num + apa_page_num + "| *\(" + apa_year_num + apa_page_num + "\))"
apa_regex = "(" + apa_author + apa_additional + "*" + apa_year + ")"

##print(apa_regex)
rxapa = re.compile(apa_regex)

rxapa_detect = re.compile(
    r"\((?:e\.g\.|see)? ?(?:[A-Za-z'`-]+(?: (?:and|&|&amp;) [A-Za-z'`-]+| et al\. ?)?,? ?\d{4}[a-z]?\;? ?)+\)")

afi_regex = r"\[((?:\d+,?-?)*)\]"
rxafi = re.compile(afi_regex)

# This is how citations will be represented inside sentence text.
# May want to change it to something that can be parsed/POS-tagged
CITATION_FORM = "<CIT ID=%s/>"

CITATION_REGEXES_JATS = {"AFI": re.compile("\[\s+<xref"),
                         "APA": re.compile("\(\s+<xref")}

CITATION_REGEXES = {"AFI": re.compile(r"\[((?:\d+,?-?)*)\]"),
                    "APA": rxapa_detect}


def guessNamesOfPlainTextAuthor(author):
    """
        Returns a dictionary with a processed author's name. Guesses which one is
        the surname
    """

    ##    res["text"]=author
    ##    if "<surname>" in author.lower():
    ##        match=rxsingleauthor.search(author)
    ##        if match:
    ##   surname=match.group(2)
    def isInitial(name):
        if name.isupper() and len(name) < 4:
            return True
        return False

    res = {}

    if "," in author:
        bits = author.split(",")
        res = {"family": bits[-1], "given": bits[0]}
        return res

    bits = author.split()

    if len(bits) > 1:
        if isInitial(bits[-1]):
            res = {"family": bits[0], "given": bits[-1]}
        else:
            res = {"family": bits[-1], "given": bits[0]}
        if len(bits) > 2:
            res["middlename"] = " ".join(bits[1:-2])
    elif len(bits) == 1:
        res["family"] = bits[0]
    else:
        pass

    return res


def normalizeAuthor(author):
    """
        Makes sure name and surname are normalized in capitalization, etc.
    """
    given = author.get("given", "").lower()
    author["given"] = given[:1].upper() + given[1:]
    family = author.get("family", "").lower()
    match = re.search(r"(" + apa_pre_family_name + r")(.*)", family)
    if match:
        family = match.group(2)
        author["family"] = match.group(1) + family[:1].upper() + family[1:]
    author["family"] = family[:1].upper() + family[1:]
    return author


def getAuthorNamesAsOneString(metadata):
    """
    Return a list of simple strings with <given name> <family name> for each author in metadata

    :param metadata: metadata dict
    :return: list of author names
    """
    author_names = []
    for author in metadata.get("authors", []):
        author_name = "{} {}".format(author.get("given", ""), author.get("family", ""))
        author_name = author_name.strip()
        author_name = author_name.lower()
        if author_name == "":
            print("ERROR: Author name is blank", author)
        else:
            author_names.append(author_name)

    return author_names


def isSameFirstAuthor(authors1, authors2):
    """
    If the first author strings of the first list and second list match, returns true

    :param authors1: list of strings
    :param authors2: list of strings
    :return: True if same first author
    """
    if len(authors1) > 0 and len(authors2) > 0 \
            and authors1[0].lower() == authors2[0].lower():
        return True
    return False


def getOverlappingAuthors(authors1, authors2):
    return list(set(authors1).intersection(set(authors2)))


def fixNumberCitationsXML(xml):
    """
        Makes [10-13] become [10 11 12 13]

        !Not yet working for some reason
    """

    def repFunc(match):
        """
            Returns a range of citations
        """
        try:
            id_1 = int(match.group(1))
            id_2 = int(match.group(3))
        except:
            # could not convert to int, they must be strings. Abort, abort
            print(match.group(0))
            return match.group(0)

        res = []
        for cnt in range(id_1, id_2 + 1):
            res.append("<xref ref-type=\"bibr\" rid=\"CIT%04d\">%d</xref>" % (cnt, cnt))
        return " ".join(res)

    def repFunc2(match):
        return jatsmultinumbercitation.sub(repFunc, match.group(0))

    xml = re.sub(r"\[.*?<xref.*?</xref>\s*?(-|--|&#x02013;)<xref.*?</xref>", repFunc2, xml, flags=re.IGNORECASE)

    return xml
    # "[<xref ref-type="bibr" rid="CIT0010">10</xref>&#x02013;<xref ref-type="bibr" rid="CIT0013">13</xref>]"


def fixNumberCitations(text):
    """
        [10-13]     -> [10 11 12 13]
        [1] [2] [3] -> [1,2,3]

        >>> fixNumberCitations("[1] [2] [3]")
        u'[1,2,3]'
        >>> fixNumberCitations("[1,2,5-7]")
        u'[1,2,5,6,7]'
    """

    def repFunc(match):
        """
            Returns a range of citations
        """
        try:
            id_1 = int(match.group(2))
            id_2 = int(match.group(3))
        except:
            # could not convert to int, they must be strings. Abort, abort
            return match.group(0)

        res = []
        for cnt in range(id_1, id_2 + 1):
            res.append(six.text_type(cnt))
        return match.group(1) + ",".join(res).strip(",") + match.group(4)

    old_text = ""
    while len(old_text) != len(text):
        old_text = text
        text = re.sub(r"(\[(?:\d+(?:-|,))*)(\d+)(?:-|--|&#x02013;)(\d+)((?:,\d+)*\])", repFunc, text,
                      flags=re.IGNORECASE)

    old_text = ""
    while len(old_text) != len(text):
        old_text = text
        text = rxseparatebrackets.sub(r"[\1,\2]", text)

    return six.text_type(text)


def detectCitationStyle(text, citation_regexes=CITATION_REGEXES, default=None):
    """
        Returns the citation style that matches the most items in the document.

        Args:
            text: string to process
            citation_regexes: dict{label:compiled_regex}
    """
    citation_count = {}
    count_non_zero = False
    for style in citation_regexes:
        count = len(citation_regexes[style].findall(text))
        if not count_non_zero:
            count_non_zero = count > 0
        citation_count[style] = count

    if count_non_zero:
        return sorted(six.iteritems(citation_count), key=lambda x: x[1], reverse=True)[0][0]

    return default


# TODO: check this fucntion actually works as it should. Make some tests.
def matchCitationWithReference(citation_data, references):
    """
        Matches an extracted in-text citation with a reference in the references

        Entirely hand-built decision tree/classifier. Hackish but it works.

        Args:
            citation_data: dict with citation data: names + year
            references: list of parsed references
        Returns:
            id of best match reference
    """

    def buildBOW(ref):
        """
            Create a quick hacked bag of words containing all the names of
            the authors
        """
        bow = [surname.lower() for surname in ref["surnames"]]
        for a in ref["authors"]:
            bow.extend(a.get("given", "").lower().split())
            bow.extend(a.get("family", "").lower().split())
        return bow

    def computeOverlap(authors, year, bow):
        """
            Returns the score of likelihood a citation points to a reference.
            Based on overlap of authors and year and an ad-hoc threshold
        """
        score = 0
        if year:
            authors.append(year)

        for i, w in enumerate(bow):
            for a in authors:
                if w.lower() == a.lower():
                    score += max(0.1, 1 - (i * 0.05))
        return score

    authors = [citation_data.get("a1", "").lower()]
    if citation_data["a2"]:
        authors.append(citation_data["a2"].lower())

    if citation_data["etal"]:
        num_authors = (2, 100)  # at least 2 authors for et al.
    else:
        if citation_data["a2"]:
            num_authors = (2, 2)  # there are exactly 2 authors
        else:
            num_authors = (1, 100)  # min 1, we don't know

    year = citation_data["year"]
    yearlen = len(six.text_type(year)) if year else 10
    found = False

    potentials = []
    if not found:
        for ref in references:
            breaking = False
            bow = buildBOW(ref)
            score = computeOverlap(authors, year, bow)
            if score >= 0.3:
                if len(ref["authors"]) >= num_authors[0] and len(ref["authors"]) <= num_authors[1]:
                    score += 0.5
                potentials.append((ref, score))

        scores = []
        for p in potentials:
            if yearlen < 4:
                diff = 2
            else:
                lev_diff = 99
                try:
                    y1 = int(year)
                    y2 = int(p[0]["year"])
                    diff = abs(y1 - y2)
                except:
                    diff = 99
                    lev_diff = levenshtein(str(year).lower(), str(p[0].get("year", "9999")).lower())
            if diff <= 2 or lev_diff <= 1:
                scores.append(p)

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if len(scores) > 0:
            ##            print("I think %s %s matches %s %s" % (authors, year, scores[0][0]["authors"], scores[0][0]["year"]),
            ##            print("with confidence %s" % scores[0][1])
            return scores[0][0]

    ##    print("Couldn't match citation data with any reference:", citation_data)

    return None


def matchCitationsWithReferences(newDocument):
    """
        Match each citation with its reference
    """
    allcitations = []
    for s in newDocument.allsentences:
        for citation_id in s.get("citations", []):
            cit = newDocument.citation_by_id[citation_id]

            if cit["ref_id"] != 0:  # the citation already has a matching reference id in the original document, use it
                match = findMatchingReferenceByOriginalId(cit["ref_id"], newDocument)
                if not match:
                    ##                        print(cit)
                    match = newDocument.matchReferenceById(cit["ref_id"])
            else:
                # attempt to guess which reference the citation should point to
                match = matchCitationWithReference(cit["original_text"], newDocument)

            if match:
                # whatever the previous case, make sure citation points to the ID of its reference
                cit["ref_id"] = match["id"]
                match["citations"].append(cit)  # add the citation to the reference's list of citations
                cit.pop("authors", "")
                cit.pop("date", "")
                cit.pop("original_text", "")
            else:
                debugAddMessage(newDocument, "notes",
                                "NO MATCH for CITATION in REFERENCES: " + cleanxml(cit["original_text"]) + ", ")
                pass


def parseCitation(intext):
    """
        FIXME: DEPRECATED

        Extract authors and date from in-text citation using plain text and regex

    """
    authors = []
    year = rxsingleyear.search(intext)
    if year: year = year.group(0)

    match = rxwtwoauthors.search(intext)
    if match:
        authors.append(match.group(1))
        authors.append(match.group(2))
    else:
        match = rxetal.search(intext)
        if match:
            authors.append(match.group(1))
        else:  # not X and X, not et al - single author
            intext = intext.replace(",", " ").replace(".", " ")
            bits = intext.split()
            authors.append(bits[0])

    return authors, year


def removeCitations(s):
    """
        FIXME: DEPRECATED

        Removes <CIT ID=x /> and <footnote>s from a string
    """
    s = re.sub(CITATION_FORM.replace("%d", r"\d+"), "", s, 0, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</?footnote.{0,11}>", " ", s, 0, re.IGNORECASE | re.DOTALL)
    return s


def removeURLs(text):
    """
        Replaces all occurences of a URL with an empty string, returns string.
    """
    return re.sub(r"(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?", "", text, flags=re.IGNORECASE)


def removeACLCitations(text):
    """
        Removes ACL-style citation tokens from text
    """
    old_text = ""
    while len(old_text) != len(text):
        old_text = six.text_type(text)
        text = re.sub(
            r"(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:, *(?:19|20)[0-9][0-9][a-g]?(?:, p.? [0-9]+)?| \((?:19|20)[0-9][0-9][a-g]?(?:, p.? [0-9]+)?\))",
            "", text)
    return text


def annotateCitationsInSentence(text, detected_style="APA"):
    """
        Returns a sentence where citations have been extracted to structured
        data (names of authors + year or number) and substituted in the sentence
        by a placeholder

        Args:
            detected_style: one of the known styles (e.g. "APA" or "AFI") for
                which we have detection regexes and extraction regexes. If None,
                it returns error
        Returns:
            tuple: (new sentence text, list of dicts with structured info)
    """
    if not detected_style:
        raise ValueError("Cannot determine the citation style")

    assert isinstance(text, six.string_types)

    if detected_style == "APA":
        return annotateCitationsAPA(text)
    elif detected_style == "AFI":
        return annotateCitationsAFI(text)


def annotateCitationsAPA(text):
    """
        Substitute an APA citation in text with a citation marker, return
        a list of dictionaries with every citation parsed

        Partly based on http://stackoverflow.com/a/10533527/363039

        >>> annotateCitationsAPA("We follow the approach of Foo et al. (2013) in doing things (Bar and Qaz, 2010; Sagae and Tsujii, 2007)")
        (u'We follow the approach of <CIT ID=1/> in doing things (<CIT ID=2/>; <CIT ID=3/>)', [{'text': 'Foo et al. (2013)', 'a1': 'Foo', 'a2': None, 'year': '2013', 'etal': True, 'type': 'APA', 'id': 0}, {'text': 'Bar and Qaz, 2010', 'a1': 'Bar', 'a2': 'Qaz', 'year': '2010', 'etal': False, 'type': 'APA', 'id': 1}, {'text': 'Sagae and Tsujii, 2007', 'a1': 'Sagae', 'a2': 'Tsujii', 'year': '2007', 'etal': False, 'type': 'APA', 'id': 2}])
    """

    extracted_citations = []

    def repFunc(match):
        """
            Substitutes each citation with a numbered placeholder, extracts
            structured data for each, adds dict to list of extracted_citations
        """
        cit_dict = {
            "type": "APA",
            "text": match.group(0),
            "id": len(extracted_citations),
            "a1": match.group(2),
            "a2": match.group(4),
            "etal": True if match.group(5) else False,
            "year": match.group(6) if match.group(5) else match.group(6)
        }

        extracted_citations.append(cit_dict)
        return CITATION_FORM % six.text_type(len(extracted_citations))

    if rxapa_detect.search(text):
        text = rxapa.sub(repFunc, text)
    return text, extracted_citations


def annotateCitationsAFI(text):
    """
        Substitute a numbered citation in text with a citation marker, return
        a list of dictionaries with every citation parsed

        >>> annotateCitationsAFI("We follow the approach of [1] in doing things [1,2,10-13]")
        (u'We follow the approach of <CIT ID=1/> in doing things <CIT ID=7/>', [{'text': u'[1]', 'num': 1, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 1, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 2, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 10, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 11, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 12, 'type': 'AFI'}, {'text': u'[1,2,10,11,12,13]', 'num': 13, 'type': 'AFI'}])
        """

    extracted_citations = []

    def repFunc(match):
        """
            Substitutes each citation with a numbered placeholder, extracts
            structured data for each, adds dict to list of extracted_citations
        """
        elements = re.sub(r"[\[\]]", "", match.group(0)).split(",")

        nums = []
        for num in elements:
            try:
                int_num = int(num)
                nums.append(int_num)
            except:
                pass
        ##                nums.append(match.group(0))

        cit_dict = {
            "type": "AFI",
            "text": match.group(0),
            "nums": nums
        }
        # each <CIT ID=%d /> token is one number more in ID
        extracted_citations.append(cit_dict)

        return CITATION_FORM % six.text_type(len(extracted_citations))

    text = fixNumberCitations(text)
    text = rxafi.sub(repFunc, text)
    return text, extracted_citations


def annotatePlainTextCitationsInSentence(sent, doc):
    """
    Adds the list of citations it finds to the sentence dict

    :param sent: sentence dictionary
    :param doc: scidoc
    :return: citations it's annotated
    """

    def replaceTempCitToken(s, temp, final):
        """
            replace temporary citation placeholder with permanent one
        """
        return re.sub(CITATION_FORM % temp, CITATION_FORM % final, s, flags=re.IGNORECASE)

    annotated_s, citations_found = annotateCitationsInSentence(sent["text"],
                                                               doc.metadata["original_citation_style"])
    annotated_citations = []

    if doc.metadata["original_citation_style"] == "APA":
        for index, citation in enumerate(citations_found):

            reference = matchCitationWithReference(citation, doc["references"])
            ##                print (citation["text"]," -> ", formatReference(reference))
            if reference:
                newCit = doc.addCitation(sent_id=sent["id"])
                newCit["ref_id"] = reference["id"]
                annotated_citations.append(newCit)
                annotated_s = replaceTempCitToken(annotated_s, index + 1, newCit["id"])
            else:
                # do something else?
                # print("Reference not found for", citation)
                annotated_s = replaceTempCitToken(annotated_s, index + 1, citation["text"])

    elif doc.metadata["original_citation_style"] == "AFI":
        for index, citation in enumerate(citations_found):
            newCit = doc.addCitation(sent_id=sent["id"])
            # TODO check this: maybe not this simple. May need matching function.
            newCit["ref_id"] = "ref" + str(int(citation["num"]) - 1)

            annotated_citations.append(newCit)
            annotated_s = replaceTempCitToken(annotated_s, index + 1, newCit["id"])

    if "citations" not in sent:
        sent["citations"] = []

    sent["citations"].extend([acit["id"] for acit in annotated_citations])
    sent["text"] = annotated_s

    # deal with many citations within characters of each other: make them know they are a cluster
    # TODO cluster citations? Store them in some other way?
    doc.countMultiCitations(sent)
    return annotated_citations, citations_found


DOCTEST = True

if DOCTEST:
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

##print fixNumberCitations("[1,2,10-13]")
##print fixNumberCitations("[1-6,10-13]")
##print fixNumberCitations("[1,2]")
##print fixNumberCitations("[1,2,5-7]")
##print fixNumberCitations("[1] [2] [3]")
##print fixNumberCitations("[1,2] [3]")
##
##print annotateCitationsAFI("We follow the approach of [1] in doing things [1,2,10-13]")
##print annotateCitationsAPA("We follow the approach of Foo et al. (2013) in doing things (Bar and Qaz, 2010; Sagae and Tsujii, 2007)")
