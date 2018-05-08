# classes to deal with showing progress and storing the results of testing
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import os, sys, codecs, math, datetime

from collections import defaultdict
from pandas import DataFrame

from proc.general_utils import (ensureDirExists, writeDictToCSV, getSafeFilename, getTimestampedFilename)
import db.corpora as cp
import six
from six.moves import range
from tqdm import tqdm


def measureScores(result_guids, match_guids, result_dict, prepend_text=""):
    """
        Method that measures the retrieval score in all the metrics for a single guid

        :param result_guids: list of guids returned by retrieval
        :param match_guids: the correct guids that should be returned
        :param result_dict: dict to which the scores will be appended

        >>> scores = {}
        >>> print(measureScores(["a", "b", "c", "d", "e"], ["a"], scores))
        {'mrr_score': 1.0, 'precision_score': 1.0, 'ndcg_score': 1.0, 'rank': 1, 'map_score': 1.0}
        >>> print(measureScores(["a", "b", "c", "d", "e"], ["b", "d"], scores))
        {'mrr_score': 0.625, 'precision_score': 0.5, 'ndcg_score': 0.7153382790366966, 'rank': 2, 'map_score': 0.6666666666666666}
    """
    assert isinstance(prepend_text, six.string_types)

    results = []
    match_guids = set(match_guids)
    found_guids = set()
    precision_at = []
    citation_multi = len(match_guids)

    idcg = 1 / math.log(2)

    for index, guid in enumerate(result_guids):
        rank = index + 1
        # if this result is one of the results we want
        if guid in match_guids and guid not in found_guids:

            # if there are several correct answers
            if citation_multi > 1:
                # if the current index+1 is up to the number of citations in the group
                if index + 1 <= citation_multi:
                    res = {"ndcg_score": 1.0,
                           "precision_score": 1.0,
                           "mrr_score": 1.0,
                           "rank": 1}
                    results.append(res)
                # if the index is beyond the number in the group
                else:
                    # it's 1 + rank to avoid log(1) == 0
                    res = {"ndcg_score": (1 / math.log(1 + rank)) / idcg,
                           "precision_score": 0.0,
                           "mrr_score": 1 / float(rank),
                           "rank": rank}
                    results.append(res)
            else:
                res = {"ndcg_score": (1 / math.log(1 + rank)) / idcg,
                       "precision_score": 1.0 if rank == 1 else 0.0,
                       "mrr_score": 1 / float(rank),
                       "rank": rank}
                results.append(res)

            found_guids.add(guid)

        precision_at.append(len(found_guids) / float(rank))
        if len(found_guids) == len(match_guids):
            break

    for guid in match_guids:
        if guid not in found_guids:
            res = {"rank": -1,
                   "precision_score": 0.0,
                   "mrr_score": 0.0,
                   "ndcg_score": 0.0}
            results.append(res)

    mrr_score = sum([res["mrr_score"] for res in results]) / len(match_guids)
    precision_score = float(sum([res["precision_score"] for res in results]) / len(match_guids))
    ndcg_score = sum([res["ndcg_score"] for res in results]) / len(match_guids)
    rank = int(sum([res["rank"] for res in results]) / len(match_guids))
    map_score = sum(precision_at) / float(len(match_guids))  # divided by the number of relevant documents to find

    result_dict[prepend_text + "mrr_score"] = mrr_score
    result_dict[prepend_text + "precision_score"] = precision_score
    result_dict[prepend_text + "ndcg_score"] = ndcg_score
    result_dict[prepend_text + "rank"] = rank
    result_dict[prepend_text + "map_score"] = map_score
    return result_dict


class ProgressIndicator(object):
    """
        Shows a basic progress indicator of items processed, left to process and
        estimated time until finishing
    """

    def __init__(self, start_counting_now=False, numitems=None, print_out=True, dot_every_xitems=None):
        self.message_text = "Processing"
        self.report_every_xseconds = 10
        self.pbar = tqdm(range(numitems))
        self.iter = self.pbar.__iter__()
        # if start_counting_now:
        self.startCounting()
        if numitems:
            self.setNumItems(numitems, print_out, dot_every_xitems)

    def startCounting(self):
        """
            Makes note of the current time.
        """
        self.start_time = datetime.datetime.now()
        self.last_report = self.start_time

    def setNumItems(self, numitems, print_out=False, dot_every_xitems=None):
        """
            Set the number of items total.

            :param numitems: number of items
            :param print_out: if True, it prints out a message when called.
            :param dot_every_xitems: if not None, it determines how often the
                progress message appears
        """
        self.numitems = numitems
        if not dot_every_xitems:
            dot_every_xitems = max(numitems / 1000, 1)
        self.dot_every_xitems = dot_every_xitems
        if print_out:
            print(self.message_text + " : ", numitems, "items...")
        self.item_counter = 0

    def showProgressReport(self, message, elapsed=1):
        """
            If a number of items or seconds has elapsed since the last report,
            it prints a progress report.
        """

        self.pbar.set_description(message)
        self.item_counter += elapsed
        next(self.iter)

        # if self.item_counter % self.dot_every_xitems == 0 \
        #         or (datetime.datetime.now() - self.last_report).total_seconds() >= self.report_every_xseconds:
        #     if datetime.datetime.now() - self.last_report >= datetime.timedelta(seconds=3):
        #         self.last_report = datetime.datetime.now()
        #         reportTimeLeft(self.item_counter, self.numitems, self.start_time, message)

    def progressReportSetNumItems(self, numitems):
        """
            Sets the number of items.
        """
        self.numitems = numitems
        self.dot_every_xitems = max(numitems / 1000, 1)
        print(self.message_text, " ", numitems, "papers...")
        self.item_counter = 0


class ResultsLogger(ProgressIndicator):
    """
        Stores the results of retrieval testing, on several metrics, and allows
        exporting results to a CSV file
    """

    def __init__(self, numitems, results_file=True, dump_straight_to_disk=True, dump_filename="results.csv",
                 message_text=None, start_counting_now=False, dot_every_xitems=None):
        ProgressIndicator.__init__(self, False, numitems=numitems, dot_every_xitems=dot_every_xitems)
        if message_text:
            self.message_text = message_text
        ##        self.mrr=defaultdict(lambda:{}) # dict for Mean Reciprocal Rank scores
        ##        self.dcg=defaultdict(lambda:{}) # Discounted Cumulative Gain
        ##        self.ndcg=defaultdict(lambda:{}) # Normalized DCG
        ##        self.precision=defaultdict(lambda:{}) # Exclusively precision score. 1 if right, 0 otherwise

        self.output_filename = dump_filename

        # stores every kind of score !TODO
        self.scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.num_data_points = defaultdict(lambda: defaultdict(lambda: 0))

        # old list to store results
        self.overall_results = []

        # new results in Pandas dataframe
        ##        self.all_results=DataFrame()

        self.text_results = []
        self.total_citations = 0
        self.numchunks = 0
        self.report_file = None
        if results_file:
            ensureDirExists(cp.Corpus.paths.output)
            self.report_file = codecs.open(os.path.join(cp.Corpus.paths.output, "report.txt"), "w")

        ##        self.citations_extra_info=["a94-1008-cit14","a94-1008-cit4", "a00-1014-cit27"]
        self.citations_extra_info = []
        self.full_citation_id = ""
        self.run_parameters = defaultdict(lambda: None)
        self.csv_columns = "file_guid citation_id doc_position query_method doc_method precision_score rank mrr_score ndcg_score az csc_ctype match_guid first_result".split()
        self.dump_straight_to_disk = dump_straight_to_disk
        if dump_straight_to_disk:
            self.startResultsDump()

    def startResultsDump(self):
        """
            Creates a dump .csv file, writes the first line
        """
        filename = getSafeFilename(getTimestampedFilename(self.output_filename, self.start_time))
        self.dump_file = codecs.open(filename, "w", "utf-8", errors="replace")

        line = u"".join([c + u"\t" for c in self.csv_columns])
        line = line.strip(u"\t")
        line += u"\n"
        self.dump_file.write(line)

    def dumpResultRow(self, row):
        """
            Writes a single result line directly to the file
        """
        line = ""
        try:
            for column_name in self.csv_columns:
                line += six.text_type(row.get(column_name, "")) + "\t"
            line = line.strip("\t")
            line += "\n"
            self.dump_file.write(line)
        except:
            ##            print "error writing: ", data_point, sys.exc_info()
            print("error writing: ", sys.exc_info()[:2])

    def computeAverageScores(self):
        """
            Computes the average accuracy of a query_method + doc_method
            combination to later choose the highest one
        """
        self.averages = defaultdict(lambda: defaultdict(lambda: {}))

        for metric in self.scores:
            for qmethod in self.scores[metric]:
                for doc_method in self.scores[metric][qmethod]:
                    data_points = self.scores[metric][qmethod][doc_method]
                    self.averages[qmethod][doc_method][metric] = data_points / float(
                        self.num_data_points[qmethod][doc_method])

    def showFinalSummary(self, overlap_in_methods=None):
        """
            Print a nice summary of everything
        """
        print()
        print("=========================================================================")
        print("Docs processed:", self.numitems)
        print("Total citations:", self.total_citations)
        print("Chunks:", self.numchunks)
        now2 = datetime.datetime.now() - self.start_time
        print("Total execution time", now2, "\n")

        for counter in cp.Corpus.global_counters:
            print(counter, ": ", cp.Corpus.global_counters[counter])
        print("Similarity:", self.run_parameters.get("similarity", None))

        overall_precision = defaultdict(lambda: 0)
        overall_mrr = defaultdict(lambda: 0)
        total_points = defaultdict(lambda: 0)

        for qmethod in self.scores["precision"]:
            for method in self.scores["precision"][qmethod]:
                overall_precision[method] += self.scores["precision"][qmethod][method] / float(self.total_citations)
                overall_mrr[method] += self.scores["mrr"][qmethod][method] / float(self.total_citations)
                total_points[method] += 1

        for method in overall_precision:
            print("Precision for", method, "%02.4f" % (overall_precision[method] / float(total_points[method])))
            print("MRR for", method, "%02.4f" % (overall_mrr[method] / float(total_points[method])))

        print("Saved to filename:", self.output_filename)

        if overlap_in_methods:
            self.computeOverlap("rank", overlap_in_methods)

    ##    print "Total rank overlap between [section_1_full_text] and [full_text_1] = ",methods_overlap,"/",total_overlap_points," = {}".format(methods_overlap / float(total_overlap_points),"2.3f")
    ##    print "Avg rank difference between [section_1_full_text] and [full_text_1] = {}".format(sum(rank_differences) / float(total_overlap_points),"2.3f")
    ##    print "Avg rank:"
    ##    for method in rank_per_method:
    ##        print method,"=",sum(rank_per_method[method])/float(len(rank_per_method[method]))

    def writeDataToCSV(self, filename=None):
        """
            Export a CSV with all the data
        """
        if not filename:
            filename = self.output_filename
        ##        self.all_results.to_csv(filename)
        if not self.dump_straight_to_disk:
            if filename:
                self.output_filename = filename
            self.output_filename = getSafeFilename(getTimestampedFilename(self.output_filename))
            writeDictToCSV(self.csv_columns, self.overall_results, filename)
        else:
            self.dump_file.close()

    def logReport(self, what):
        """
            Throw anything at this, its representation will be written in the report file
        """
        if self.report_file:
            if isinstance(what, six.string_types):
                try:
                    self.report_file.write(what + "\n")
                except:
                    self.report_file.write(what.__repr__() + "\n")
            else:
                self.report_file.write(what.__repr__() + "\n")

    def addQMethod(self, qmethod):
        """
            A method that just deals with creating a dict for each Qmethod. Should really be using defaultdicts
        """
        pass

    def measureScoreAndLog(self, retrieved_docs, citation_multi, result_dict):
        """
            file_guid, retrieved_docs, right_citation, doc_position, qmethod, method, query, az, cfc
            guid,retrieved,m,doc_position,qmethod,method, queries[qmethod]["text"], az, cfc

            Computes all scores and position document was ranked in
        """
        # docs must be already ordered in descending order
        guids = []
        for doc in retrieved_docs:
            guid = doc[1]["guid"]
            if guid not in guids:
                guids.append(guid)

        measureScores(guids, result_dict["match_guids"], result_dict)

        self.logReport("Rank: " + str(result_dict["rank"]))
        self.logReport("Correct: " + str(result_dict["match_guids"]) + " Retrieved: " + guids[0])
        result_dict["first_result"] = guids[0]
        self.addResolutionResultDict(result_dict)
        return result_dict

    def addResolutionResultDict(self, result_dict):
        """
            Adds a data point about a resolved citation, successful or not
        """
        ##        data_point=(file_guid,right_citation["cit"]["id"], doc_position, qmethod, method, mrr_score, ndcg_score, precision_score, right_citation["match_guid"])

        if self.dump_straight_to_disk:
            self.dumpResultRow(result_dict)
        else:
            # !TODO Proper do something about the results logging. Use ResultStorer?
            try:
                self.overall_results.append(result_dict)
            except:
                print("OUT OF MEMORY ERROR!")
                print("Dumping results so far to file...")
                if self.output_filename:
                    self.scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
                    self.writeDataToCSV()
                    self.overall_results = []

        ##        self.all_results=self.all_results.append(DataFrame(result_dict))

        qmethod = result_dict["query_method"]
        method = result_dict["doc_method"]

        # these aren't necessary: it could be rebuilt from the overall_results at the end, but it's faster
        self.scores["mrr"][qmethod][method] += result_dict["mrr_score"]
        self.scores["ndcg"][qmethod][method] += result_dict["ndcg_score"]
        self.scores["precision"][qmethod][method] += result_dict["precision_score"]
        self.scores["rank"][qmethod][method] += result_dict["rank"]

        # this is to compute averages
        self.num_data_points[qmethod][method] += 1

    def computeOverlap(self, overlap_in, overlap_between):
        """
            Shows the overlap between methods
        """
        data = DataFrame(self.overall_results)
        group = data.groupby(["file_guid", "citation_id"])

        all_overlaps = []
        for a, b in group:
            numitems = b.shape[0]
            results = []
            for i in range(numitems):
                doc_method = b.iloc[i]["doc_method"]
                rank = b.iloc[i][overlap_in]
                if doc_method in overlap_between:
                    results.append(rank)

            this_one = 1
            for index in range(len(results) - 1):
                if results[index] != results[index + 1]:
                    this_one = 0
                    break

            all_overlaps.append(this_one)

        print("Overlap between", overlap_between, ": %02.4f" % (sum(all_overlaps) / float(len(all_overlaps))))


DOCTEST = False

if DOCTEST:
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


def main():
    # scores = {}
    # measureScores(["a", "b", "c", "d", "e"], ["b", "d"], scores)
    # measureScores(["a", "b", "c", "d", "e"], ["a"], scores)
    # print(scores)
    pass


if __name__ == '__main__':
    main()
