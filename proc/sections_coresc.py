# Match sections
#
# Copyright (C) 2014 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


def matchGenericSection(header, prev):
    """
        Returns what generic section you're in
    """
    # word overlap scores?
    sections={"Introduction":"introduction",
    "Abstract":"abstract",
    "Conclusion":"conclusion concluding remarks conclusions",
    "Future work": "future future future",
    "Data":"data corpus corpora statistics dataset datasets corpus corpus",
    "Related work":"previous work related works comparison",
    "Results/Discussion":"discussion discussions results",
    "Problem":"problem task",
    "Motivation":"background motivation problem formalism idea example investigating",
    "Methodology":"methods method methodology feature features algorithm heuristic our approach preprocessing model training measure measures framework",
    "Implementation":"implementation overview",
    "Experiments":"experiment experiments experimental experimentation",
    "Evaluation":"evaluation evaluating judgement analysis test performance",
    "Acknowledgements": "acknowledgements"
    }

    for s in sections:
        sections[s]=sections[s].split()

    scores={"":0}
    words=re.sub(r"[\d\.,-:;\s]{1,30}"," ",header.lower()).strip().split()
    wdict={}
    for w in words:
        wdict[w]=wdict.get(w,0)+1

    for w in wdict:
        for s in sections:
            for w2 in sections[s]:
                if w==w2:
                    scores[s]=scores.get(s,0)+wdict[w]

    res=sorted(scores.iteritems(),key=lambda x:x[1],reverse=True)[0]
    return res[0]

def matchCoreSC(section,prev):
    """
        Returns the CoreSC equivalent of the current section
    """
    CSC=['Hypothesis', 'Motivation', 'Goal', 'Object', 'Background', 'Method', 'Experiment', 'Model',
 'Observation', 'Result', 'Conclusion']
    header=section["header"].lower()
    match1=re.match(r"\s*\d\.\d(\.\d)*.*",header)
    if match1: # is subheader
        #can ignore
        pass

    for c in [c.lower() for c in CSC]:
        if c in header:
            print "yay",c+"!"

    print header


def main():
    pass

if __name__ == '__main__':
    main()
