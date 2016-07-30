# Generate a HTML for a file, highlighting each sentence with its CoreSC type
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>
# For license information, see LICENSE.TXT

import os, re

import minerva.db.corpora as cp
from minerva.proc.general_utils import writeFileText
from minerva.proc.nlp_functions import tokenizeText
from minerva.scidoc.render_content import SciDocRenderer
from minerva.scidoc.reference_formatting import formatCitation

from collections import defaultdict

CURRENT_CITATION=""

def extraAttributes(s,attributes, glob):
    """
        Adds extra class attributes to each sentence to format them differently
        according to their class
    """
    if s.get("type","") != "s":
        return attributes

    attributes["class"]="sentence"

    az_type=s.get("az","")
    if az_type != "":
        attributes["class"]= attributes.get("class","") + " AZ_"+az_type

    csc_type=s.get("csc_type","")
    if csc_type != "":
        attributes["class"]=attributes.get("class","") + " CSC_"+csc_type

    attributes["class"]=attributes["class"].strip()

def referenceFormatting(text, glob, ref):
    """
        Function to be passed to JsonDoc.prettyPrintDocumentHTML()

        Just adds a span around each reference, distinguishing between in-collection
        references and not. Behaviour depends on CSS/JS
    """
    match=cp.Corpus.matcher.matchReference(ref)
    reftype="reference"
    if match:
        reftype+=" in-collection"
    res="<span class='"+reftype+"'>"+text+"</span>"
    return res

def citationFormatting(text, match):
    """
        Changes the class of all citations not pointing to the current citation
        to avoid highlighting them
    """
    global CURRENT_CITATION

    class_match=re.search(CURRENT_CITATION, text, re.IGNORECASE)
    if not class_match:
        text=re.sub(r"class=\"citation\"","class=\"citation_ignore\"", text, flags=re.IGNORECASE)
    return text

def padWithHTML(html):
    """
        Adds <html> tags and includes required stylesheet/JS
    """
    result="""<html><head><meta charset="utf-8" /> <link href='scidocview.css' rel='stylesheet'>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>

    <script>
        csc_dict={
        	 "Hyp":"Hypothesis. A statement not yet confirmed rather than a factual statement",
        	 "Mot":"Motivation. The reasons behind an investigation",
        	 "Bac":"Background. Generally accepted background knowledge and previous work",
        	 "Goa":"Goal. A target state of the investigation where intended discoveries are made",
        	 "Obj":"Object-New. An entity which is a product or main theme of the investigation",
        	 "Met":"Method. Means by which authors seek to achieve a goal of the investigation OR A method mentioned pertaining to previous work",
        	 "Exp":"Experiment. An experimental method",
        	 "Mod":"Model. A statement about a theoretical model or framework",
        	 "Obs":"Observation. The data/phenomena recorded in an investigation",
        	 "Res":"Result. Factual statements about the outputs of an investigation",
        	 "Con":"Conclusion. Statements inferred from observations &amp; results relating to research hypothesis",
        };
    </script>

    </head><body>
    <div id="overlay_box">

    </div>
    """+html+"""
    <script type="text/javascript" language="javascript">
        $('.sentence').hover(function (evt) {
            var text=$(evt.target).attr("class").replace("sentence ","");
            var class_text=text.replace("_","");
            var class_text=text.replace("CSC_","");

            var desc="<b>"+class_text+"</b> "+csc_dict[class_text];

            $("#overlay_box").html(desc);
            $("#overlay_box").show();
            $("#overlay_box").css({
                top: evt.pageY,
                left: evt.pageX
            });
        });

        $('.sentence').mouseout(function()
        {
            $("#overlay_box").hide();
        });
    </script>

    </body></html>"""
    return result

def explainInFileZoning(guid):
    """
        Given a guid, generates a HTML visualization where sentences are tagged
        with their class for the contents of that file
    """
    doc=cp.Corpus.loadSciDoc(guid)
    renderer=SciDocRenderer(doc)
    html=renderer.prettyPrintDocumentHTML(True,True, False, extra_attribute_function=extraAttributes, wrap_with_HTML_tags=False,
        reference_formatting_function=referenceFormatting)
    html=padWithHTML(html)
    writeFileText(html,os.path.join(cp.Corpus.paths.output,guid+"_text_zoning.html"))


def trimDocToRelevantBits(doc, guid):
    """
        Selects only the paragraphs in the document where the paper identified by
        ``guid`` is cited, returns them as a modified scidoc.

        Warning: modifies doc in-place.
    """
    linked_ref=None
    for ref in doc.references:
        ref_guid=None
        if "guid" in ref:
            ref_guid=ref["guid"]
        else:
            match=cp.Corpus.matcher.matchReference(ref)
            if match:
                ref_guid=match["guid"]

        if ref_guid==guid:
            linked_ref=ref
            break

    if not linked_ref:
        # couldn't match the relevant citation, return
        return

    new_sect=doc.addSection("root", "")
    para=doc.addParagraph(new_sect["id"])
    sent=doc.addSentence(para["id"],"MATCHED REFERENCE: %s" % formatCitation(linked_ref))
    content=[new_sect, para, sent]
    to_add=[]

    for cit_id in linked_ref["citations"]:
        cit=doc.citation_by_id[cit_id]
        parent_p=None
        if "parent_p" in cit:
            para=cit["parent_p"]
        else:
            parent_s=cit.get("parent_s",None) or cit["parent"]
            sent=doc.element_by_id[parent_s]
            para=doc.element_by_id[sent["parent"]]
        para["parent"]=new_sect["id"]
        to_add.append(para)
        to_add.extend([doc.element_by_id[s] for s in para["content"]])

    to_add=list(set([element["id"] for element in to_add]))
    to_add=[doc.element_by_id[id] for id in to_add]
    for element in to_add:
        if element.get("type","") == "p":
            new_sect["content"].append(element["id"])

    content.extend(to_add)
    doc["content"]=content
    doc.updateContentLists()


def explainAnchorTextZoning(guid, max_inlinks=10, use_full_text=False):
    """
        This generates a clipping collection file, including all the citation
        contexts of other files to this file
    """
    meta=cp.Corpus.getMetadataByGUID(guid)
    all_html=["""<h1 class="title">%s</h1><span>Inlink context summary for %s</span>""" % (meta["title"],formatCitation(meta))]
    global CURRENT_CITATION
    CURRENT_CITATION=re.escape(formatCitation(meta))

    for index, link in enumerate(meta["inlinks"]):
        if index == max_inlinks:
            break
        print("Processing anchor text from %s" % link)
        doc=cp.Corpus.loadSciDoc(link)

        if not use_full_text:
            trimDocToRelevantBits(doc, guid)

        renderer=SciDocRenderer(doc)
        html=renderer.prettyPrintDocumentHTML(
            formatspans=True,
            include_bibliography=use_full_text,
            wrap_with_HTML_tags=False,
            extra_attribute_function=extraAttributes,
            citation_formatting_function=citationFormatting,
            reference_formatting_function=referenceFormatting)
        all_html.append(html)

    html=padWithHTML(" ".join(all_html))
    writeFileText(html,os.path.join(cp.Corpus.paths.output,guid+"_ilc_zoning.html"))

def connectToCorpus():
    """
    """
    cp.useElasticCorpus()
    from minerva.squad.config import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.Corpus.connectCorpus(r"g:\nlp\phd\pmc_coresc", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

def main():
    connectToCorpus()
##    guid="df8c8824-1784-46f1-b621-cc6e5aca0dad"
    guid="d7aceeb7-f1a0-4301-9b66-e5f32c476aac"

    explainInFileZoning(guid)
    explainAnchorTextZoning(guid, 30)
    pass

if __name__ == '__main__':
    main()
