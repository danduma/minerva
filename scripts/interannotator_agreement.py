import os
import json
import re
from proc.stopword_proc import getStopwords
from scripts.manual_annotation import readAnnotationFile, generateAnnotationFile
from collections import Counter


def export_first_to_second(queries_filename):
    q_dir = os.path.dirname(queries_filename)
    stopwords = getStopwords(q_dir)


def cleanKeywords(filename, existing_annot):
    """
    Cleans manual annotation by removing punctuation and corpus-specific stopwords.

    :param filename:
    :param existing_annot:
    :return:
    """
    stopwords = getStopwords(filename)

    for annot_id in existing_annot:
        annot = existing_annot[annot_id]
        all_kws = []
        for kp in annot["keywords"]:
            kws = re.split("\W+", kp.strip())
            kws = [kw.lower() for kw in kws if kw.lower() not in stopwords and len(kw) > 2]
            all_kws.extend(kws)

        annot["keywords"] = all_kws

    return existing_annot


def generateFileForSecondAnnotator(input_filename, output_filename):
    """
    Creates an annotation file for 2nd annotator, only with the contexts that the first annotator annotated.

    :param input_filename: filename of existing first annotator output *_annot.txt
    :return: None
    """
    annot_2nd_file = output_filename

    _, existing_annot = readAnnotationFile(input_filename)
    existing_annot = {k: existing_annot[k] for k in existing_annot if len(existing_annot[k]["keywords"]) > 0}

    writeAnnotatedFile(annot_2nd_file, existing_annot)

def generateFirstAnnotatorFile(input_filename, output_filename):
    """
    Creates an annotation file from 1st annotator, only with the contexts that the first annotator annotated.

    :param input_filename: filename of existing first annotator output *_annot.txt
    :return: None
    """
    annot_2nd_file = output_filename

    _, existing_annot = readAnnotationFile(input_filename)
    existing_annot = {k: existing_annot[k] for k in existing_annot if len(existing_annot[k]["keywords"]) > 0}

    writeAnnotatedFile(annot_2nd_file, existing_annot, write_first_keywords=True)


def writeAnnotatedFile(filename, existing_annot, write_first_keywords=False, limit=100):
    """
    Wtrites the annotation file.

    :param filename: filename to write to
    :param existing_annot: a dictionary with the existing annotations
    :param write_first_keywords: if True, will export the annotations from the first annotator
    :return: None
    """
    f = open(filename, "w")
    done_queries = {}
    index = 0

    for annot_id in existing_annot:
        if index >= limit:
            break

        annot = existing_annot[annot_id]
        if len(annot["keywords"]) == 0:
            continue

        if annot_id in done_queries:
            continue

        kws = " ".join(annot["keywords"])

        f.write("#ID: " + annot["id"] + "\n")
        f.write("#NUM: %d\n\n" % index)
        f.write("#WORDS: " + annot["words"] + "\n\n")
        if write_first_keywords:
            f.write("#KEYWORDS: %s \n\n\n" % kws)
        else:
            f.write("#KEYWORDS: %s \n\n\n" % "")
        done_queries[annot_id] = True
        index += 1

    print("Wrote ", index, "contexts")


def computeInterAnnotatorAgreement(file_1st, file_2nd):
    """
    Computes P, R and F1 scores for 2 annotators

    :param file_1st:
    :param file_2nd:
    :return:
    """
    _, annots1 = readAnnotationFile(file_1st)
    _, annots2 = readAnnotationFile(file_2nd)

    annots1 = cleanKeywords(file_1st, annots1)
    annots2 = cleanKeywords(file_2nd, annots2)

    avg_list = []

    for annot in annots1:
        if annot not in annots2:
            continue

        if len(annots2[annot]["keywords"]) < 1:
            continue

        annots1[annot]["keywords2"] = annots2[annot]["keywords"]
        s1 = set(annots1[annot]["keywords"])
        s2 = set(annots1[annot]["keywords2"])

        tp = len(s1.intersection(s2))
        fp = len(s2.difference(s1))
        fn = len(s1.difference(s2))
        if (tp + fp) == 0 or tp + fn == 0:
            pass
        try:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = (2 * p * r) / (p + r)
        except:
            print("arg")

        annots1[annot]["p"] = p
        annots1[annot]["r"] = r
        annots1[annot]["f1"] = f1

        avg_list.append((p, r, f1))

    print("total results:", len(avg_list))
    print("p: %.3f" % (sum([x[0] for x in avg_list]) / len(avg_list)))
    print("r: %.3f" % (sum([x[1] for x in avg_list]) / len(avg_list)))
    print("f1: %.3f" % (sum([x[2] for x in avg_list]) / len(avg_list)))
    return annots1, avg_list


def generateSecondAnnotatorFiles():
    generateFileForSecondAnnotator(
        "/Users/masterman/Dropbox/PhD/thesis/data/aac/precomputed_queries_new1k_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_aac.txt")

    generateFileForSecondAnnotator(
        "/Users/masterman/Dropbox/PhD/thesis/data/pmc/precomputed_queries_test_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_pmc.txt")


def generateFirstAnnotatorFiles():
    generateFirstAnnotatorFile(
        "/Users/masterman/Dropbox/PhD/thesis/data/aac/precomputed_queries_new1k_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_aac_annotated_1st.txt",
    )

    generateFirstAnnotatorFile(
        "/Users/masterman/Dropbox/PhD/thesis/data/pmc/precomputed_queries_test_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_pmc_annotated_1st.txt")


def computeAgreement():
    computeInterAnnotatorAgreement(
        "/Users/masterman/Dropbox/PhD/thesis/data/aac/precomputed_queries_new1k_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_aac_annotated.txt",
    )
    computeInterAnnotatorAgreement(
        "/Users/masterman/Dropbox/PhD/thesis/data/pmc/precomputed_queries_test_annot.txt",
        "/Users/masterman/Dropbox/PhD/thesis/data/keywords_pmc_annotated.txt",
    )


def main():
    generateFirstAnnotatorFiles()
    # generateSecondAnnotatorFiles()
    # computeAgreement()


if __name__ == '__main__':
    main()
