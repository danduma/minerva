# various NLP related functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import re
from string import punctuation
from nltk import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer

AZ_ZONES_LIST=["AIM","BAS","BKG","CTR","OTH","OWN","TXT"]
CORESC_LIST=["Hyp","Mot","Bac","Goa","Obj","Met","Exp","Mod","Obs","Res","Con"]
RANDOM_ZONES_7=["RND7_"+str(x) for x in range(7)]
RANDOM_ZONES_11=["RND11_"+str(x) for x in range(11)]


basic_stopwords="a an and or not the that for with on in off after over".split()

stopwords_list=["a","able","about","above","according","accordingly","across",
"actually","after","afterwards","again","against","all","allow","allows","almost",
"alone","along","already","also","although","always","am","among","amongst","an",
"and","another","any","anybody","anyhow","anyone","anything","anyway","anyways",
"anywhere","apart","appear","appreciate","appropriate","are","aren't","around",
"as","aside","ask","asking","associated","at","available","away","awfully","be",
"became","because","become","becomes","becoming","been","before","beforehand",
"behind","being","believe","below","beside","besides","best","better","between",
"beyond","both","brief","but","by","c'mon","c's","came","can","can't","cannot",
"cant","cause","causes","certain","certainly","changes","clearly","co","com",
"come","comes","concerning","consequently","consider","considering","contain",
"containing","contains","corresponding","could","couldn't","course","currently",
"definitely","described","despite","did","didn't","different","do","does",
"doesn't","doing","don't","done","down","downwards","during","each","edu",
"eg","eight","either","else","elsewhere","enough","entirely","especially",
"et","etc","even","ever","every","everybody","everyone","everything","everywhere",
"ex","exactly","example","except","far","few","fifth","first","five","followed",
"following","follows","for","former","formerly","forth","four","from","further",
"furthermore","get","gets","getting","given","gives","go","goes","going","gone",
"got","gotten","greetings","had","hadn't","happens","hardly","has","hasn't","have",
"haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter",
"hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither",
"hopefully","how","howbeit","however","i'd","i'll","i'm","i've","ie","if","ignored",
"immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates",
"inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll",
"it's","its","itself","just","keep","keeps","kept","know","known","knows",
"last","lately","later","latter","latterly","least","less","lest","let","let's",
"like","liked","likely","little","look","looking","looks","ltd","mainly","many",
"may","maybe","me","mean","meanwhile","merely","might","more","moreover","most",
"mostly","much","must","my","myself","name","namely","nd","near","nearly",
"necessary","need","needs","neither","never","nevertheless","new","next","nine",
"no","nobody","non","none","noone","nor","normally","not","nothing","novel",
"now","nowhere","obviously","of","off","often","oh","ok","okay","old","on",
"once","one","ones","only","onto","or","other","others","otherwise","ought",
"our","ours","ourselves","out","outside","over","overall","own","particular",
"particularly","per","perhaps","placed","please","plus","possible","presumably",
"probably","provides","que","quite","qv","rather","rd","re","really","reasonably",
"regarding","regardless","regards","relatively","respectively","right","said",
"same","saw","say","saying","says","second","secondly","see","seeing","seem",
"seemed","seeming","seems","seen","self","selves","sensible","sent","serious",
"seriously","seven","several","shall","she","should","shouldn't","since","six",
"so","some","somebody","somehow","someone","something","sometime","sometimes",
"somewhat","somewhere","soon","sorry","specified","specify","specifying","still",
"sub","such","sup","sure","t's","take","taken","tell","tends","th","than","thank",
"thanks","thanx","that","that's","thats","the","their","theirs","them","themselves",
"then","thence","there","there's","thereafter","thereby","therefore","therein",
"theres","thereupon","these","they","they'd","they'll","they're","they've",
"think","third","this","thorough","thoroughly","those","though","three","through",
"throughout","thru","thus","to","together","too","took","toward","towards",
"tried","tries","truly","try","trying","twice","two","un","under","unfortunately",
"unless","unlikely","until","unto","up","upon","us","use","used","useful","uses",
"using","usually","value","various","very","via","viz","vs","want","wants","was",
"wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were",
"weren't","what","what's","whatever","when","whence","whenever","where","where's",
"whereafter","whereas","whereby","wherein","whereupon","wherever","whether",
"which","while","whither","who","who's","whoever","whole","whom","whose","why",
"will","willing","wish","with","within","without","won't","wonder","would",
"wouldn't","yes","yet","you","you'd","you'll","you're","you've","your","yours",
"yourself","yourselves"]

stopwords=["~", "the", "and", "or", "not", "of", "to", "from", "by", "with", "a", "an"]
stopwords.extend(punctuation)
CIT_MARKER="__cit"
PAR_MARKER="__par__"
BR_MARKER="__br__"

CITATION_PLACEHOLDER=CIT_MARKER
ESTIMATED_AVERAGE_WORD_LENGTH=10

##from org.tartarus.snowball.ext import PorterStemmer;
global_stemmer=SnowballStemmer("english")

USING_STEMMING=False

rx_sentences_to_add=re.compile(r"(?:(?:(\d)up)_?(?:(\d)down)?(_withinpara)?)|(paragraph)|(1only)", re.IGNORECASE)

# helper functions
def removeCitations(s):
    """
        Removes <CIT ID=x /> from a string
        and <footnote>s
        and <eqn>s
    """
    s=re.sub(r"<cit.*?/>","", s, 0, flags=re.DOTALL|re.IGNORECASE)
    s=re.sub(r"</?footnote.{0,11}>"," ",s, 0, re.IGNORECASE|re.DOTALL)
    return s

def sentenceSplit(text):
    """
        Returns sentences from a text. At present using NLTK Punkt tokenizer
    """
##    try:
##        res=sent_tokenize(text)
##    except UnicodeDecodeError as e:
##        print ("UNICODE DECODE ERROR: ",e.message)
##        res=[text]
##    return res
    return sent_tokenize(text)

def tokenizeText(text, no_stemming=False):
    """
        Doesn't remove stopwords. Automatically stems if USING_STEMMING is True.

        Abstracts over how the tokenizing is actually done

        Returns: list of tokens (strings is assumed, may be dict)
    """
    tokens=word_tokenize(text)
    if USING_STEMMING and not no_stemming:
        return stemTokens(tokens)
    else:
        return tokens

def removeStopwords(tokens, stopwords=stopwords):
    """
        Returns the tokens in the list that are not in the list of stopwords
    """
    return [t for t in tokens if t not in stopwords]

def tokenizeTextAndRemoveStopwords(text, stopwords=stopwords):
    """
        Removes stopwords. Abstracts over how the tokenizing is actually done

        Returns: list of tokens (strings is assumed, may be dict) minus stopwords
    """
    tokens = tokenizeText(text)
    tokens = removeStopwords(tokens, stopwords)
    return tokens

def replaceCitationsWithPlaceholders(text):
    """
        Substitutes all <CIT> elements in the sentence with __cit__
    """
    return re.sub(r"<CIT ID=(.*?)\s?/>",CITATION_PLACEHOLDER, text)

def replaceCitationTokensForParsing(text):
    """
        Substitutes all <CIT> elements in the sentence with __cit__
    """
    return re.sub(r"<CIT ID=\w*?(\d+)\s?/>",r"__cit\1", text)


def getCitationNumberFromToken(text):
    """
        Returns the citation number from a single-token citation
    """
    match=re.search(r"__cit(\d+)",text,flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))

def getFirstNumberFromString(text):
    """
        Returns any number found in a string
    """
    match=re.search(r"(\d+)",text,flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))

def formatSentenceForIndexing(s, no_stemming=False):
    """
        Fixes all the contents of the sentence, returns a sentence that's easy
        to index for IR
    """
    text=s["text"]
    text=removeCitations(text)
    tokens=tokenizeText(text, no_stemming)
    tokens=removeStopwords(tokens)
    text=unTokenize(tokens)
    return text

def selectSentencesToAdd(docfrom,cit,param):
    """
        Returns a list of sentence IDs (strings) to include based on the [param]

        :param docfrom: SciDoc we are extracting from
        :param cit: citation dict
        :param param: ["paragraph", "1only", "[n]up_[n]down[_crosspar]"]
        :returns list of sentence IDs
        :rtpye list
    """
    sent=docfrom.element_by_id[cit["parent_s"]]
    para=docfrom.element_by_id[sent["parent"]]

    match=rx_sentences_to_add.search(param)
    assert(match)

    context={"ilc_AZ_"+zone:"" for zone in AZ_ZONES_LIST}
    for zone in CORESC_LIST:
        context["ilc_CSC_"+zone]=""
    to_add=[]

    if match.group(4): # paragraph
        to_add=para["content"]
    elif match.group(5): # 1only
        to_add=[cit["parent_s"]]
    else:               #1up_1down, etc
        sent_up=int(match.group(1)) if match.group(1) else 0
        sent_down=int(match.group(2)) if match.group(2) else 0
        within_par=int(match.group(3) != None)


        if not within_par:
            all_sentence_ids=[s["id"] for s in docfrom.allsentences]
        else:
            all_sentence_ids=para["content"]

        index=all_sentence_ids.index(cit["parent_s"])
        if index > 0:
            to_add.extend(all_sentence_ids[index-sent_up:index])

        to_add.append(cit["parent_s"])

        if index < len(docfrom.allsentences)-1:
            to_add.extend(all_sentence_ids[index+1:index+1+sent_down])

    return to_add


def removePunctuation(tokens):
    return [token for token in tokens if token not in punctuation]


def getDictOfTokenCounts(tokens):
    """
        Returns a dictionary of token counts

        Args:
            tokens: list of tokens (strings)
    """
    res={}
    for token in tokens:
        res[token]=res.get(token,0)+1
    return res

def stemTokens(tokens):
    """
        Stems a list of tokens
    """
    res=[]
    for token in tokens:
##        res.append(token)
##        new_token=global_stemmer.stem(token)
##        if new_token != token:
##            res.append(new_token)
        if token not in [CITATION_PLACEHOLDER, CIT_MARKER]:
            res.append(global_stemmer.stem(token))
        else:
            res.append(token)
    return res

def unTokenize(tokens):
    """
        Returns a string from a list of tokens
    """
    return u" ".join(tokens)


def stemText(text):
    """
        Tokenizes, stems, untokenizes text
    """
    return unTokenize(stemTokens(tokenizeText(text)))


def main():
##    print stemTokens(tokenizeText("this text is a test1 123test __CIT__, ,, !!"))
    pass

if __name__ == '__main__':
    main()
