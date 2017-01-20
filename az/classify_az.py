# AZ classification
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from minerva.scidoc.scidoc import SciDoc
from minerva.scidoc.xmlformats.azscixml import *

import collections, random, nltk
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier

from az_features import buildAZFeatureSetForDoc

import glob
import os
import cPickle

# For maxent training
MIN_LL_DELTA=0.0005 # minimum increment per step
MAX_ITER=30 # max number of iterations for training


def buildGlobalFeatureset(input_mask,output_file):
    """
        Creates a list of all sentences in the collections, their features and class,
        for classifier training/testing
    """
    doc_list=glob.glob(input_mask)
    global_featureset=[]

    for filename in doc_list:
        doc=SciDoc(filename)
        featureset=buildAZFeatureSetForDoc(doc)
        global_featureset.extend(featureset)

    cPickle.dump(global_featureset,file(output_file,"w"))
    return global_featureset

#===============================
#       TESTING CODE
#===============================
drive="C"
global_featureset_filename=drive+r":\nlp\phd\raz\converted_scidoc_files\global_featureset.pickle"
input_mask=drive+r":\nlp\phd\raz\converted_scidoc_files\*.json"


def runTestAZ(rebuild=False):
    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeatureset(input_mask,global_featureset_filename)
    else:
        global_featureset=cPickle.load(file(global_featureset_filename))

    train_set=global_featureset[:len(global_featureset)/10]
    test_set=global_featureset[len(global_featureset)/10:]

    print("Training classifier")
    classifier = nltk.MaxentClassifier.train(train_set, min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)
##    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Accuracy:",nltk.classify.accuracy(classifier, test_set))

    classified=[classifier.classify(x[0]) for x in test_set]

    cm = nltk.ConfusionMatrix([x[1] for x in test_set], classified)
    print(cm.pp(sort_by_count=True, show_percents=True, truncate=9))

def runKFoldCrossValidation(rebuild=False, folds=3):
    """
        Tests the classifier with K-fold cross-validation

    """
    from sklearn import cross_validation

    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeatureset(input_mask,global_featureset_filename)
    else:
        global_featureset=cPickle.load(file(global_featureset_filename))

    cv = cross_validation.KFold(len(global_featureset), n_folds=folds, indices=True, shuffle=False, random_state=None, k=None)

    accuracies=[]

    print("Beginning",folds,"-fold cross-validation")

    for traincv, testcv in cv:
##        print "Training classifier"
##        print traincv,testcv
##        print traincv[0],":",traincv[-1]
##        print testcv[0],":",testcv[-1]

        train_set=[global_featureset[i] for i in traincv]
        test_set=[global_featureset[i] for i in testcv]

        # select type of classifier here
##        classifier = nltk.NaiveBayesClassifier.train(train_set)
        classifier = nltk.MaxentClassifier.train(global_featureset[traincv[0]:traincv[len(traincv)-1]], min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)

        accuracy=nltk.classify.util.accuracy(classifier, test_set)
        print('accuracy:', accuracy)
        accuracies.append(accuracy)
        classified=[classifier.classify(x[0]) for x in test_set]
        cm = nltk.ConfusionMatrix([x[1] for x in test_set], classified)
        print(cm.pp(sort_by_count=True, show_percents=True, truncate=9))

    print ("average accuracy:",sum(accuracies)/float(len(accuracies)))



def trainAZfullCorpus(filename,rebuild=False):
    """
        Trains and saves an AZ classifier for the full corpus, saves it in filename
    """
    if rebuild:
        print("Rebuilding global featureset")
        global_featureset=buildGlobalFeatureset(input_mask,global_featureset_filename)
    else:
        global_featureset=cPickle.load(file(global_featureset_filename))

    classifier = nltk.MaxentClassifier.train(global_featureset, min_lldelta=MIN_LL_DELTA,max_iter=MAX_ITER)
    cPickle.dump(classifier,open(filename,"w"))


def main():
##    convertAnnotToSciDoc(r"g:\NLP\PhD\raz\input\*.annot",r"C:\NLP\PhD\raz\converted_scidoc_files")

##    doc=SciDoc()
##    doc=loadAZSciXML(r"g:\nlp\phd\raz\input\9405001.annot")
##    featureset=buildFeatureSetForDoc(doc)

##    runTestAZ(True)
    runKFoldCrossValidation(True,5)
##    trainAZfullCorpus("trained_az_classifier.pickle",True)

##    doc.saveToFile(r"c:\nlp\raz\converted_scidoc_files\9405001.json")


if __name__ == '__main__':
    main()
