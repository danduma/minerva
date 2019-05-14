# functions for Citation Function Classification
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from __future__ import absolute_import
from __future__ import print_function

import glob
import os

import nltk.classify.util
import nltk.metrics
import six.moves.cPickle

from scidoc.xmlformats.azscixml import *
from .az_features import *

# For maxent training
MIN_LL_DELTA=0.0005 # minimum increment per step
MAX_ITER=30 # max number of iterations for training


def testTypesContained(container):
    """
        Will return a set with all the types contained in a container (list or dict)
    """
    def recurseTypes(cont2, typeset):
        typeset.add(type(element))
        if isinstance(cont2,list):
            elements=cont2
        elif isinstance(element, dict):
            elements=list(element.values())
        else:
            return

        for element in elements:
            recurseTypes(element,typeset)

    types=set()
    recurseTypes(container,types)
    return types

def convertAnnotToSciDoc(input_mask,output_dir):
    """
        Given a mask "c:\bla\*.annot" it will load each file and save it in
        output_dir as a SciDoc .JSON
    """
    output_dir=ensureTrailingBackslash(output_dir)
    for filename in glob.glob(input_mask)[2:]:
        print("Converting",filename)
        doc=loadAZSciXML(filename)
        fn=os.path.basename(filename)
        doc.saveToFile(output_dir+os.path.splitext(fn)[0]+".json")


def buildGlobalFeaturesetCFC(input_mask,output_file):
    """
        Creates a list of all citations in the collections, their features and class,
        for classifier training/testing
    """
    doc_list=glob.glob(input_mask)
    global_featureset=[]

    for filename in doc_list:
        doc=SciDoc(filename)
        featureset=buildCFCFeaturesetForDoc(doc)
        global_featureset.extend(featureset)

    six.moves.cPickle.dump(global_featureset,open(output_file,"w"))
    return global_featureset

#===============================
#       TESTING CODE
#===============================
global_featureset_filename=r"G:\NLP\PhD\cfc\converted_files\global_featureset.pickle"
input_mask=r"G:\NLP\PhD\cfc\converted_files\*.json"

def runTestCFC(rebuild=False):

    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeaturesetCFC(input_mask,global_featureset_filename)
    else:
        global_featureset=six.moves.cPickle.load(open(global_featureset_filename))

    train_set=global_featureset[:len(global_featureset)/10]
    test_set=global_featureset[len(global_featureset)/10:]

    print("Training classifier")
    classifier = nltk.MaxentClassifier.train(train_set, min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)
##    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Accuracy:",nltk.classify.accuracy(classifier, test_set))

    classified=[classifier.classify(x[0]) for x in test_set]

    cm = nltk.ConfusionMatrix([x[1] for x in test_set], classified)
    print((cm.pp(sort_by_count=True, show_percents=True, truncate=9)))

def runKFoldCrossValidation(rebuild=False, folds=3):
    from sklearn import model_selection

    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeaturesetCFC(input_mask,global_featureset_filename)
    else:
        global_featureset=six.moves.cPickle.load(open(global_featureset_filename))

    cv = model_selection.KFold(n_splits=folds, shuffle=False, random_state=None)
    cv = zip(cv.split(global_featureset))
    accuracies=[]

    print("Beginning",folds,"-fold cross-validation")

    for traincv, testcv in cv:
##        print "Training classifier"
##        print traincv,testcv
##        print traincv[0],":",traincv[-1]
##        print testcv[0],":",testcv[-1]

        train_set=global_featureset[traincv[0]:traincv[-1]]
        test_set=global_featureset[testcv[0]:testcv[-1]]

##        classifier = nltk.NaiveBayesClassifier.train(train_set)
        classifier = nltk.MaxentClassifier.train(global_featureset[traincv[0]:traincv[len(traincv)-1]], min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)

        accuracy=nltk.classify.util.accuracy(classifier, test_set)
        print('accuracy:', accuracy)
        accuracies.append(accuracy)

    print("average accuracy:",sum(accuracies)/float(len(accuracies)))
    classified=[classifier.classify(x[0]) for x in test_set]
    cm = nltk.ConfusionMatrix([x[1] for x in test_set], classified)
    print((cm.pp(sort_by_count=True, show_percents=True, truncate=9)))


def trainCFCfullCorpus(filename,rebuild=False):
    """
        Trains and saves a CFC classifier for the full corpus, saves it in filename
    """
    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeaturesetCFC(input_mask,global_featureset_filename)
    else:
        global_featureset=six.moves.cPickle.load(open(global_featureset_filename))

    classifier = nltk.MaxentClassifier.train(global_featureset, min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)
    six.moves.cPickle.dump(classifier,open(filename,"w"))


def main():
##    convertAnnotToSciDoc(r"G:\NLP\PhD\cfc\2006_paper_training\*.cfc-scixml",r"G:\NLP\PhD\cfc\converted_files")

##    runTestCFC(True)
##    runKFoldCrossValidation(True,5)
    trainCFCfullCorpus("trained_cfc_classifier.pickle",True)
##    doc.saveToFile(r"c:\nlp\raz\converted_scidoc_files\9405001.json")


if __name__ == '__main__':
    main()
