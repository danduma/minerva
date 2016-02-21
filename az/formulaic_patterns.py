# implements Teufel's formulaic patterns with RegEx
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json, re, os, inspect

class formulaicPattern:
    def __init__(self):
        self.patterns={}
        self.agents={}
        self.concepts={}
        self.compiled_patterns={}
        self.compiled_agents={}

        self.filepath=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.loadData()
        self.makeAllRegExes()

    def loadData(self):
        """
            Loads formulaic patterns and concept lexicon
        """
        f=open(os.path.join(self.filepath,"formulaic_patterns.json"),"r")
        self.patterns=json.load(f)
        f.close()

        f=open(os.path.join(self.filepath,"concept_lexicon.json"),"r")
        self.concepts=json.load(f)
        f.close()

        f=open(os.path.join(self.filepath,"agent_patterns.json"),"r")
        self.agents=json.load(f)
        f.close()


    def makeRegEx(self,pattern):
        """
            Takes one pattern, returns one compiled regex using
            the concept lexicon
        """
        def replConcept(match):
            concept=match.group(1)
            if self.concepts.has_key(concept):
                if len(self.concepts[concept])==1:
                    return self.concepts[concept][0]
                else:
                    return "("+"|".join(self.concepts[concept]).strip("|")+")"
            print match.group(0)
            return "!!NOT FOUND!!"

        rx=pattern.replace(" ","\s+")
        rx=rx.replace("*","") # we don't need the important flag
        # not implementing POS yet
        rx=rx.replace("JJ","\w+")
        rx=rx.replace("NN","\w+")
        rx=rx.replace("RB","\w+")

        rx=rx.replace("CITE","__CITE__")

        rx=re.sub(r"@(\w+)",replConcept,rx,0,re.IGNORECASE)
##        print rx
        comp_rx=re.compile(rx,re.IGNORECASE)
        return comp_rx

    def makeAllRegExes(self):
        """
            Goes through the patterns, creating compiled regexes from each using
            the concept lexicon
        """

        for pattern_type in self.patterns:
            for pattern in self.patterns[pattern_type]:
                crx=self.makeRegEx(pattern)
                self.compiled_patterns[pattern_type]=self.compiled_patterns.get(pattern_type,[])
                self.compiled_patterns[pattern_type].append(crx)

        for agent_type in self.agents:
            for pattern in self.agents[agent_type ]:
                crx=self.makeRegEx(pattern)
                self.compiled_agents[pattern_type]=self.compiled_agents.get(agent_type,[])
                self.compiled_agents[pattern_type].append(crx)


    def extractFeatures(self,sent,features,agents=False):
        """
            Will return a list of the pattern types that are matched in a sentence
        """
##        result=[]
        collection=self.compiled_agents if agents else self.compiled_patterns

        for pattern_type in collection:
            for crx in collection[pattern_type]:
                if crx.search(sent):
##                    result.append(pattern_type)
                    features["F_"+pattern_type]=True
                    continue
##        return result

def processConceptLexicon():
    f=open("concept_lexicon.txt","r")
    result={}
    for line in f:
        bits=line.split(":")
        name=bits[0]
        content=[x.strip() for x in bits[1].split(",")]
        result[name]=content
##        print name,content
    f2=open("concept_lexicon.json","w")
    json.dump(result,f2,indent=3)

def main():
##    loadFormulaicPatterns()
    processConceptLexicon()
    form=formulaicPattern()

    pass

if __name__ == '__main__':
    main()
