# functions to extract content and generate bag-of-words for indexing
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import re, copy

from collections import defaultdict, OrderedDict

from nlp_functions import tokenizeText, stemText, stopwords, stemTokens, \
CITATION_PLACEHOLDER, unTokenize, ESTIMATED_AVERAGE_WORD_LENGTH, removeCitations, \
PAR_MARKER, CIT_MARKER, BR_MARKER, AZ_ZONES_LIST, CORESC_LIST

from general_utils import getListAnyway

import minerva.db.corpora as corpora

USING_STEMMING=True

# this is for adding fields to a document in Lucene. These fields are not to be indexed
FIELDS_TO_IGNORE=["left_start","right_end","params","guid_from","year_from"]

def formatSentenceForIndexing(s, no_stemming=False):
    """
        Fixes all the contents of the sentence, returns a sentence that's easy
        to index for IR
    """
    text=s["text"]
    text=re.sub(r"<CIT ID=(.*?)\s?/>",CITATION_PLACEHOLDER, text)
    # ! IMPORTANT, uncomment the next line to enable stemming
    if USING_STEMMING and not no_stemming:
        text=stemText(text)
    return text

def findCitationInFullText(cit,doctext):
    """
        To avoid seeking the citation again for every query extraction method
    """
    return re.search(r"<cit\sid="+str(cit["id"])+r"\s*?/>",doctext, flags=re.DOTALL|re.IGNORECASE)


def joinCitationContext(leftwords,rightwords, extract_dict):
    """
        Joins the words to the left and to the right of a citation into one string
    """
    assert isinstance(leftwords,list)
    assert isinstance(rightwords,list)
    allwords=[]
    allwords.extend(leftwords)
    allwords.extend(rightwords)
    allwords=[token for token in allwords if token.lower() not in stopwords]
    if USING_STEMMING:
        allwords=stemTokens(allwords)
##    extract_dict["text"]=unTokenize(allwords)
    # Always convert in-text citations to placeholders
    extract_dict["text"]=re.sub(r"<CIT ID=(.*?)\s?/>",CITATION_PLACEHOLDER, unTokenize(allwords))
    return extract_dict

def getDictOfTestingMethods(methods):
    """
        Make a simple dictionary of {method_10:{method details}}

        new: prepare for using a single Lucene index with fields for the parameters
    """
    res=OrderedDict()
    for method in methods:
        for parameter in methods[method]["parameters"]:
            if methods[method]["type"] in ["standard_multi","inlink_context"]:
                addon="_"+str(parameter)
                indexName=method+addon
                res[indexName]=copy.deepcopy(methods[method])
                res[indexName]["method"]=method
                res[indexName]["parameter"]=parameter
                res[indexName]["index_filename"]= methods[method]["index"]+addon
                res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"]
##                res[indexName]["index_field"]=str(parameter)
            elif methods[method]["type"] in ["ilc_mashup"]:
                for ilc_parameter in methods[method]["ilc_parameters"]:
                    addon="_"+str(parameter)+"_"+str(ilc_parameter)
                    indexName=method+addon
                    res[indexName]=copy.deepcopy(methods[method])
                    res[indexName]["method"]=method
                    res[indexName]["parameter"]=parameter
                    res[indexName]["ilc_parameter"]=ilc_parameter
                    res[indexName]["index_filename"]=methods[method]["index"]+addon
                    res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"]
##                    res[indexName]["index_field"]=str(parameter)+"_"+str(ilc_parameter)
            elif methods[method]["type"] in ["annotated_boost"]:
                for runtime_parameter in methods[method]["runtime_parameters"]:
                    indexName=method+"_"+str(parameter)+"_"+runtime_parameter
                    res[indexName]=copy.deepcopy(methods[method])
                    res[indexName]["method"]=method
                    res[indexName]["parameter"]=parameter
                    res[indexName]["runtime_parameters"]= methods[method]["runtime_parameters"][runtime_parameter]
##                    res[indexName]["index_filename"]=methods[method]["index"]+"_"+str(parameter)
                    res[indexName]["index_filename"]=methods[method]["index"]+"_"+str(parameter)
            elif methods[method]["type"] in ["ilc_annotated_boost"]:
                for ilc_parameter in methods[method]["ilc_parameters"]:
                    for runtime_parameter in methods[method]["runtime_parameters"]:
                        indexName=method+"_"+str(ilc_parameter)+"_"+runtime_parameter
                        res[indexName]=copy.deepcopy(methods[method])
                        res[indexName]["method"]=method
                        res[indexName]["parameter"]=parameter
                        res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"][runtime_parameter]
                        res[indexName]["ilc_parameter"]=ilc_parameter
                        res[indexName]["index_filename"]=methods[method]["index"]+"_"+str(parameter)+"_"+str(ilc_parameter)
##                    res[indexName]["index_field"]=str(parameter)+"_"+str(ilc_parameter)
    return res


##prebuild_indeces={
##"passage":{"type":"standard_multi", "bow_methods":[("passage",[150,175,200,250,300,350,400,450])]},
##"inlink_context":{"type":"standard_multi", "bow_methods":[("inlink_context",[5, 10, 15, 20, 30, 40, 50])]},
####    "az_annotated":{"type":"standard_multi", "bow_methods":[("az_annotated",[1])]},
####    "section_annotated":{"type":"standard_multi", "bow_methods":[("az_annotated",[1])]},
##
##"ilc_section_annotated":{"type":"ilc_mashup", "mashup_method":"section_annotated", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##"ilc_passage":{"type":"ilc_mashup", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50], "parameters":[250,300,350]},
##"ilc_az_annotated":{"type":"ilc_mashup", "mashup_method":"az_annotated", "ilc_parameters":[10,20,30], "parameters":[1]},
##}


def getDictOfLuceneIndeces(prebuild_indices):
    """
        Make a simple dictionary of {method_10:{method details}}

    """
    res=OrderedDict()
    for method in prebuild_indices:
        for parameter in prebuild_indices[method]["parameters"]:
            if prebuild_indices[method]["type"] in ["standard_multi","inlink_context"]:
                indexName=method+"_"+str(parameter)
                res[indexName]=copy.deepcopy(prebuild_indices[method])
                res[indexName]["method"]=prebuild_indices[method].get("bow_name",method)
                res[indexName]["parameter"]=parameter
                res[indexName]["index_filename"]=indexName
                res[indexName]["index_field"]=str(parameter)
                res[indexName]["options"]=prebuild_indices[method].get("options",{}) # options apply to all parameters
            elif prebuild_indices[method]["type"] in ["ilc_mashup"]:
                for ilc_parameter in prebuild_indices[method]["ilc_parameters"]:
                    indexName=method+"_"+str(parameter)+"_"+str(ilc_parameter)
                    res[indexName]=copy.deepcopy(prebuild_indices[method])
                    res[indexName]["method"]=prebuild_indices[method].get("bow_name",method)
                    res[indexName]["parameter"]=parameter
                    res[indexName]["ilc_method"]=prebuild_indices[method]["ilc_method"]
                    res[indexName]["ilc_parameter"]=ilc_parameter
                    res[indexName]["index_filename"]=indexName
                    res[indexName]["index_field"]=str(parameter)+"_"+str(ilc_parameter)
                    res[indexName]["options"]=prebuild_indices[method].get("options",{}) # options apply to all parameters
    return res


def processContext(match,doctext, wleft, wright):
    """
        Optimize the extraction of context by limiting the amount of characters and pre-tokenizing
    """
    left_start=max(0,match.start()-(wleft*ESTIMATED_AVERAGE_WORD_LENGTH))
    right_end=match.end()+(wright*ESTIMATED_AVERAGE_WORD_LENGTH)

    left=doctext[left_start:match.start()] # tokenize!
    left=tokenizeText(removeCitations(left))

    right=doctext[match.end():right_end] # tokenize!
    right=tokenizeText(removeCitations(right))

    return {"left":left,"right":right,"left_start":left_start,"right_end":right_end}

def identifyReferenceLinkIndex(docfrom, linkto):
    """
        Returns the id in the references list of a document that a
        "linked to" document matches
    """
    for ref in docfrom["references"]:
        match=corpora.Corpus.matchReferenceInIndex(ref)
        if match and match["guid"]==linkto["guid"]:
            return ref["id"]
    return None


def addFieldsFromDicts(d1,d2):
    """
        Concatenates the strings in dictionary d2 to those in d1,
        saves in d1
    """
    for key in d2:
        if key not in FIELDS_TO_IGNORE:
            # !TODO temporary fix to lack of separation of sentences
            d1[key]=d1.get(key,"")+" "+d2[key].replace("."," . ")
##            d1[key]=d1.get(key,"")+" "+d2[key]

def joinTogetherContextExcluding(context_list, exclude_files=[], max_year=None, exclude_authors=[], full_corpus=False):
    """
        Context_list= list of dicts{"guid_from":<guid>,"year_from":<year>,"text":<text>}}
        Concatenate all contexts, excluding the one for exclude_file
        Returns a dictionary that joins together the context as selected in one BOW
        {"inlink_context"="bla bla"}
    """
    # !TODO deal with AZ-tagged context
    # dealt with it already?
    res=defaultdict(lambda:"")

    assert isinstance(context_list,list)
    for cont in context_list:
        if cont["guid_from"] not in exclude_files:
            # !TODO Exclude authors from inlink_context and test
            if not max_year:
                addFieldsFromDicts(res,cont)
            elif int(cont["year_from"]) <= max_year:
                addFieldsFromDicts(res,cont)
        else:
##            print "Excluded", cont["guid_from"]
            pass

    # Important! RENAMING "text" field to "inlink_context" for all BOWs thus created
    res["inlink_context"]=res.pop("text") if res.has_key("text") else ""

    return res

def getFieldSpecialTestName(fieldname, test_guid):
    """
        Returns the name of a "Special" test field
    """
    return fieldname+"_special_"+test_guid


def filterInlinkContext(context_list, exclude_list=[], max_year=None, exclude_authors=[], full_corpus=False):
    """
        Deals with the difference between a per-test-file-index and a full corpus index,
        creating the appropriate
    """
    if full_corpus:
        #TODO FIX METHOD!
        bow={}
        for exclude_guid in exclude_list:
            bow_temp=joinTogetherContextExcluding(context_list,[exclude_guid], max_year)
            bow[getFieldSpecialTestName(method,exclude_guid)]=bow_temp[method]
        bows=[bow]
    else:
        bows=[joinTogetherContextExcluding(context_list, exclude_list, max_year)]

    return bows

def getOutlinkContextWindowAroundCitation(match, doctext, wleft=20, wright=20):
    """
        Default method, up to x words left, x words right
    """

    context=processContext(match,doctext, wleft, wright)

    leftwords=context["left"][-wleft:]
    rightwords=context["right"][:wright]

    left_start=match.start()-sum([len(context["left"][-x])+1 for x in range(min(len(context["left"]),wleft))])
    right_end=match.end()+sum([len(context["right"][x])+1 for x in range(min(len(context["right"]),wright))])
    extract_dict={"left_start":left_start,"right_end":right_end,"params":(wleft,wright)}
    return joinCitationContext(leftwords,rightwords,extract_dict)


def getOutlinkContextWindowAroundCitationMulti(match, doctext, parameters=[(20,20)], options={"jump_paragraphs":True}):
    """
        Default method, up to x words left, x words right
        returns a dict {"left_start","right_end"}
    """

    context=processContext(match,doctext, max([x[0] for x in parameters]), max([x[1] for x in parameters]))

    res=[]

    for words in parameters:
        leftwords=context["left"][-words[0]:]
        rightwords=context["right"][:words[1]]
        left_start=match.start()-sum([len(context["left"][-x])+1 for x in range(min(len(context["left"]),words[0]))])
        right_end=match.end()+sum([len(context["right"][x])+1 for x in range(min(len(context["right"]),words[1]))])
        extract_dict={"left_start":left_start,"right_end":right_end,"params":words}
        res.append(joinCitationContext(leftwords,rightwords,extract_dict))

    return res

def getOutlinkContextSentences(docfrom, cit, param, separate_by_tag=None, dict_key="text"):
    """
        Returns one or more sentences put into a same bag of words, separated by

    """
    from az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST

    sent=docfrom.element_by_id[cit["parent_s"]]
    para=docfrom.element_by_id[sent["parent"]]

    if separate_by_tag=="az":
        context={"ilc_AZ_"+zone:"" for zone in AZ_ZONES_LIST}
    else:
        context={dict_key:""}

    to_add=[]
    if param=="paragraph":
        to_add=para["content"]
    elif param=="1up_1down":
        index=para["content"].index(cit["parent_s"])
        if index > 0:
            to_add.append(para["content"][index-1])

        to_add.append(cit["parent_s"])

        if index < len(para["content"])-1:
            to_add.append(para["content"][index+1])
    elif param=="1up":
        index=para["content"].index(cit["parent_s"])
        if index > 0:
            to_add.append(para["content"][index-1])

        to_add.append(cit["parent_s"])
    elif param=="1only":
        to_add=[cit["parent_s"]]

    for sent_id in to_add:
        sent=docfrom.element_by_id[sent_id]
        text=formatSentenceForIndexing(sent)
        if separate_by_tag=="az":
            context["ilc_AZ_"+sent["az"]]+=text+" "
        elif separate_by_tag=="csc":
            context["ilc_CSC_"+sent["csc_type"]]+=text+" "
        else:
            context[dict_key]+=text+" "

    context["params"]=param
    return context

def getOutlinkContextSentencesMulti(docfrom, cit, params, separate_by_tag=None, dict_key="text"):
    """
        Returns one or more sentences put into a same bag of words, separated by

    """
    res=[]
    for param in params:
        res.append(getOutlinkContextSentences(docfrom, cit, param, separate_by_tag=None, dict_key="text"))
    return res

def getOutlinkContextFancy(match, doctext, wleft=40, wright=40, stopbr=False):
    """
        Fancier extraction of words around citation
    """
    def modTextExtractContext(text):
        """
            Change linke breaks, paragraph breaks and citations to tokens
        """
        text=re.sub(r"<cit\sid=.{1,5}\s*?/>"," "+CIT_MARKER+" ",text, 0, re.IGNORECASE|re.DOTALL)
        text=re.sub(r"</?footnote.{0,11}>"," ",text, 0, re.IGNORECASE|re.DOTALL)
        text=re.sub(r"\n\n"," "+PAR_MARKER+" ",text)
        text=re.sub(r"\n"," "+BR_MARKER+" ",text)
        return text

    left=doctext[max(0,match.start()-(wleft*ESTIMATED_AVERAGE_WORD_LENGTH)):match.start()] # tokenize!
    left=modTextExtractContext(left)
    left=tokenizeText(left)

    right=doctext[match.end():match.end()+(wright*ESTIMATED_AVERAGE_WORD_LENGTH)] # tokenize!
    right=modTextExtractContext(right)
    right=tokenizeText(right)

    leftwords=[]
    left=[x for x in reversed(left)]
    for w in left[:wleft]:
        new=[]
        if w==PAR_MARKER:    # paragraph break
            break
        if w==CIT_MARKER:    # citation
            break
        if w==BR_MARKER:    # line break
            if stopbr:
                print "break: br"
                break
            else:
                continue
        else:
            new.append(w)
            new.extend(leftwords)
            leftwords=new

    rightwords=[]
    for w in right[:wright]:
        if w==PAR_MARKER:    # paragraph break
            print "break: paragraph"
            break
        if w==CIT_MARKER:    # citation
            print "break: citation"
            break
        if w==BR_MARKER:    # line break
            if stopbr:
                print "break: br"
                break
            else:
                continue
        else:
            rightwords.append(w)

##    print "Q Fancy:",
    return joinCitationContext(leftwords, rightwords, {})


def getOutlinkContextFancyMulti(match, doctext, parameters=[(20,20)], maxwords=20, options={"jump_paragraphs":True}):
    """
        Fancier method

        returns a list = [[BOW from param1], [BOW from param2]...]
    """

    left=doctext[max(0,match.start()-(maxwords*ESTIMATED_AVERAGE_WORD_LENGTH)):match.start()] # tokenize!
    left=tokenizeText(removeCitations(left))

    right=doctext[match.end():match.end()+(maxwords*ESTIMATED_AVERAGE_WORD_LENGTH)] # tokenize!
    right=tokenizeText(removeCitations(right))

    res=[]

    for words in parameters:
        leftwords=left[-words[0]:]
        rightwords=right[:words[1]]
        res.append(joinCitationContext(leftwords,rightwords))

    return res



# prebuilt files functions


#===============================================================================
# getDocBOW methods = generate VSM representation of document
#===============================================================================
def generateDocBOWInlinkContext(doc_incoming, parameters, doctext=None):
    """
        Create a BOW from all the inlink contexts of a given document
    """
    all_contexts={}
    for param in parameters:
        all_contexts[param]=[]

    doc_metadata=corpora.Corpus.getMetadataByGUID(doc_incoming["metadata"]["guid"])
    print "Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]), "incoming links"

    for inlink_guid in doc_metadata["inlinks"]:
        # loads from cache if exists, XML otherwise
        docfrom=corpora.Corpus.loadSciDoc(inlink_guid)
        # important! the doctext here has to be that of the docfrom, NOT doc_incoming
        doctext=docfrom.getFullDocumentText()
        ref_id=identifyReferenceLinkIndex(docfrom, doc_incoming["metadata"])

        print "Document with incoming citation links loaded:", docfrom["metadata"]["filename"]

        for param in parameters:
            all_contexts[param]=all_contexts.get(param,[])

            for cit in docfrom["citations"]:
                if cit["ref_id"] == ref_id:
                    # need a list of citations for each outlink, to extract context from each
                    match=findCitationInFullText(cit,doctext)
                    if not match:
                        print "Weird! can't find citation in text"
                        continue

                    context=getOutlinkContextWindowAroundCitation(match, doctext, param, param)
                    # code below: store list of dicts with filename, BOW, so I can filter out same file later
                    context["guid_from"]=docfrom["metadata"]["guid"]
                    context["year_from"]=docfrom["metadata"]["year"]
                    all_contexts[param].append(context)

    #   this bit of code makes every entry a list for multiple representations from each document
##    for c in all_contexts:
##        all_contexts[c]=[all_contexts[c]]
    return all_contexts


def generateDocBOW_ILC_Annotated(doc_incoming, parameters, doctext=None, full_paragraph=True):
    """
        Create a BOW from all the inlink contexts of a given document
        Extracts sentences around the citation, annotated with their AZ

        parameters = {"full_paragraph":True,"sent_left":1, "sent_right":1}?
    """
    all_contexts=defaultdict(lambda:[])
    for param in parameters:
        all_contexts[param]=[]

    doc_metadata=corpora.Corpus.getMetadataByGUID(doc_incoming["metadata"]["guid"])
    print "Building VSM representations for ", doc_metadata["guid"], ":", len(doc_metadata["inlinks"]), "incoming links"

    for inlink_guid in doc_metadata["inlinks"]:
        # loads from cache if exists, XML otherwise
        docfrom=corpora.Corpus.loadSciDoc(inlink_guid)
        corpora.Corpus.annotateDoc(docfrom,["AZ"])

        # important! the doctext here has to be that of the docfrom, NOT doc_incoming
        doctext=docfrom.getFullDocumentText()
        ref_id=identifyReferenceLinkIndex(docfrom, doc_incoming["metadata"])

        print "Document with incoming citation links loaded:", docfrom["metadata"]["filename"]

        for param in parameters:
            citations=[cit for cit in docfrom["citations"] if cit["ref_id"] == ref_id]
            for cit in citations:
                # need a list of citations for each outlink, to extract context from each
                sent=docfrom.element_by_id[cit["parent_s"]]
                para=docfrom.element_by_id[sent["parent"]]

                context={"ilc_AZ_"+zone:"" for zone in AZ_ZONES_LIST}
                for zone in CORESC_LIST:
                    context["ilc_CSC_"+zone]=""
                to_add=[]
                if param=="paragraph":
                    to_add=para["content"]
                elif param=="1up_1down":
                    index=para["content"].index(cit["parent_s"])
                    if index > 0:
                        to_add.append(para["content"][index-1])

                    to_add.append(cit["parent_s"])

                    if index < len(para["content"])-1:
                        to_add.append(para["content"][index+1])
                elif param=="1up":
                    index=para["content"].index(cit["parent_s"])
                    if index > 0:
                        to_add.append(para["content"][index-1])

                    to_add.append(cit["parent_s"])
                elif param=="1only":
                    to_add=[cit["parent_s"]]

                for sent_id in to_add:
                    sent=docfrom.element_by_id[sent_id]
                    text=formatSentenceForIndexing(sent)
                    context["ilc_AZ_"+sent["az"]]+=" "+text
                    if "csc_type" not in sent:
                        sent["csc_type"]="Bac"
                    context["ilc_CSC_"+sent["csc_type"]]+=" "+text

                context["guid_from"]=docfrom["metadata"]["guid"]
                context["year_from"]=docfrom["metadata"]["year"]
                all_contexts[param].append(context)

    #   this bit of code makes every entry a list for multiple representations from each document
##    for c in all_contexts:
##        all_contexts[c]=[all_contexts[c]]
    return all_contexts


def mashupBOWinlinkMethods(doc_incoming, exclude_files, max_year, method_params, full_corpus=False):
    """
        Returns BOWs ready to add to indeces of different parameters for inlink_context
    """

    ilc_bow=corpora.Corpus.loadPrebuiltBOW(doc_incoming["guid"], method_params["ilc_method"], method_params["ilc_parameter"])
    if ilc_bow is None:
        print "Prebuilt BOW not found for: inlink_context", method_params["ilc_method"] + method_params["ilc_parameter"]
        return None

    # this is a dict representing the context and eventually a new "document" to be added to the index
    ilc_bow=filterInlinkContext(ilc_bow, exclude_files, max_year, full_corpus=full_corpus)

    # for some reason this shouldn't be a list
    ilc_bow=ilc_bow[0]

    bow_method1=getListAnyway(corpora.Corpus.loadPrebuiltBOW(doc_incoming["guid"],method_params["mashup_method"],method_params["parameter"]))

    res=[]
    for passage in bow_method1:
        addFieldsFromDicts(passage,ilc_bow)
##        addDocBOWFullTextField(doc,res,doctext)
        res.append(passage)
    return res

def mashupAnnotatedBOWinlinkMethods(doc_incoming, exclude_files, max_year, method1, param1, inlink_parameter):
    """
        Returns BOWs ready to add to indeces of different parameters for inlink_context
    """

    ilc_bow=corpora.Corpus.loadPrebuiltBOW(doc_incoming["guid"], "inlink_context", inlink_parameter)
    if ilc_bow is None:
        print "Prebuilt BOW not found for: inlink_context", inlink_parameter
        return None

    # this is a dict representing the context and eventually a new "document" to be added to the index
    ilc_bow=joinTogetherContextExcluding(ilc_bow,exclude_files,max_year)

    bow_method1=getListAnyway(corpora.Corpus.loadPrebuiltBOW(doc_incoming["guid"],method1,param1))

    res=[]
    for passage in bow_method1:
        addFieldsFromDicts(passage,ilc_bow)
##        addDocBOWFullTextField(doc,res,doctext)

        res.append(passage)
    return res


def getDocBOWInlinkContextCache(doc, exclude_files, parameters, doctext=None):
    """
        Same as getDocBOWInlinkContext, uses prebuilt inlink contexts from disk cache,
        rebuilds them where not available
    """
    newparam=[]
    result={}
##    self_id=doc["guid"] # avoid including in the BOW context from the very file
##    max_date=doc["year"]

    for param in parameters:

        if not corpora.Corpus.prebuiltFileExists(doc["guid"],"inlink_context", param):
            newparam.append(param)
        else:
##            print "Loading prebuilt inlink_context BOW from cache for", doc_incoming["filename"], param
            bow=corpora.Corpus.loadPrebuiltBOW(doc["guid"],"inlink_context", param)
            # bow[0] would be necessary if the function returns a list of BOWs for each param

            result[param]=[{"text":joinTogetherContextExcluding(bow,exclude_files,)}]

    if len(newparam) > 0:
        print "New parameters I don't have prebuilt BOWs for:",newparam
        new=generateDocBOWInlinkContext(doc_incoming, newparam, doctext)
        for param in new:
            result[param]=[{"text":joinTogetherContextExcluding(new[param],exclude_files)}]

    return result


def getDocBOWfull(doc, parameters=None, doctext=None):
    """
        Get BOW for document using full text minus references and section titles
    """
    if not doctext:
        doctext=doc.getFullDocumentText(doc)
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    new_doc={"text":unTokenize(tokens)}
    return {1:[new_doc]} # all functions must take a list of parameters and return dict[parameter]=list of BOWs

def getDocBOWpassagesMulti(doc, parameters=[100], doctext=None):
    """
        Get BOW for document using full text minus references and section titles

        returns multiple BOWs in a dictionary where the keys are the parameters
    """

    if not doctext:
        doctext=doc.getFullDocumentText(doc, headers=True)

    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)

    res={}

    for param in parameters:
        res[param]=[]

        for i in xrange(0, len(tokens), param):
            bow={"text":unTokenize(tokens[i:i+param])}
            res[param].append(bow)

        for i in xrange(param/2, len(tokens), param):
            bow={"text":unTokenize(tokens[i:i+param])}
            res[param].append(bow)

    return res

##def getDocBOWfullHeaders(doc, parameters=None, doctext=None):
##    """
##        Get BOW for document using full text minus references
##    """
##    if not doctext:
##        doctext=doc.getFullDocumentText(doc, headers=True)
##    doctext=removeCitations(doctext).lower()
##    tokens=tokenizeText(doctext)
##    return {1:[unTokenize(tokens)]}


def getDocBOWTitleAbstract(doc, parameters=None, doctext=None):
    """
        Get BOW for document using only title and abstract
    """
    paragraphs=[]
    doctext=doc["metadata"]["title"]+". "
    if len(doc.allsections) > 0:
        try:
            doctext+=doc.getSectionText(doc.allsections[0],False)
        except:
            doctext+=u"<UNICODE ERROR>"
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    return {1:[{"text":unTokenize(tokens)}]}

def getDocBOWannotated(doc, parameters=None, doctext=None, keys=["az","csc_type"]):
    """
        Get BOW for document with AZ
    """
    res=defaultdict(lambda:"")
    for sentence in doc.allsentences:
        text=formatSentenceForIndexing(sentence)
        for key in keys:
            if sentence.has_key(key):
                res[sentence[key]]+=text
    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}

def getDocBOWrandomZoning(doc, parameters=None, doctext=None, keys=["az","csc_type"]):
    """
        Get BOW for document with AZ
    """
    res=defaultdict(lambda:"")
    for sentence in doc.allsentences:
        text=formatSentenceForIndexing(sentence)
        res[random.choice(AZ_ZONES_LIST)]+=text
        res[random.choice(CORESC_LIST)]+=text

    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}


def getDocBOWannotatedSections(doc, parameters=None, doctext=None):
    """
        Returns a dict where each key should be a field for the document in
        the Lucene index
        returns {"title","abstract","text"}
    """
    paragraphs=[]
    res={}
    res["title"]=doc["metadata"]["title"]
    res["abstract"]=""

    if len(doc.allsections) > 0 :
        try:
            res["abstract"]=doc.getSectionText(doc.allsections[0],False)
            res["abstract"]==removeCitations(res["abstract"]).lower()
        except:
            res["abstract"]=u"<UNICODE ERROR>"

    paper_text=""
    for index in range(1,len(doc.allsections),1):
        paper_text+=doc.getSectionText(doc.allsections[index],False)

    res["text"]=removeCitations(paper_text).lower()
    addDocBOWFullTextField(doc,res,doctext)
    return {1:[res]}

def addDocBOWFullTextField(doc,res_dict,doctext=None):
    """
        Adds the _full_text field
    """
    if not doctext:
        doctext=doc.getFullDocumentText(doc)
    doctext=removeCitations(doctext).lower()
    tokens=tokenizeText(doctext)
    res_dict["_full_text"]=unTokenize(tokens)

def main():

    pass

if __name__ == '__main__':
    main()
