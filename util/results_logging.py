#-------------------------------------------------------------------------------
# Name:        results_logging
# Purpose:     classes to deal with showing progress and storing the results of
#               testing
#
# Author:      dd
#
# Created:     05/02/2015
# Copyright:   (c) dd 2015
#-------------------------------------------------------------------------------

import codecs, math
from general_utils import *
import corpora
from collections import defaultdict
from pandas import DataFrame, Series

class progressIndicator:
    """
        Shows a basic progress indicator of items processed, left to process and
        estimated time until finishing
    """
    def __init__(self,start_counting_now=False):
        if start_counting_now:
            self.startCounting()

    def startCounting(self):
        self.start_time=datetime.datetime.now()
        self.last_report=self.start_time

    def setNumItems(self, numitems, print_out=True, dot_every_xitems=None):
        self.numitems=numitems
        if not dot_every_xitems:
            dot_every_xitems=max(numitems / 1000,1)
        self.dot_every_xitems=dot_every_xitems
        if print_out:
            print "Processing: ", numitems, "items..."
        self.item_counter=0

    def showProgressReport(self, message, elapsed=1):
        self.item_counter+=elapsed
        if self.item_counter % self.dot_every_xitems ==0:
            if datetime.datetime.now()-self.last_report >= datetime.timedelta(seconds=3):
                self.last_report=datetime.datetime.now()
                reportTimeLeft(self.item_counter,self.numitems, self.start_time, message)

    def progressReportSetNumItems(self, numitems):
        self.numitems=numitems
        self.dot_every_xitems=max(numitems/ 1000,1)
        print "Processing ", numitems, "papers..."
        self.item_counter=0

class resultsLogger (progressIndicator):
    """
        Stores the results of retrieval testing, on several metrics, and allows
        exporting results to a CSV file
    """
    def __init__(self, results_file=True, dump_straight_to_disk=True,dump_filename="results.csv"):
        progressIndicator.__init__(self, False)
        self.text=""
##        self.mrr=defaultdict(lambda:{}) # dict for Mean Reciprocal Rank scores
##        self.dcg=defaultdict(lambda:{}) # Discounted Cumulative Gain
##        self.ndcg=defaultdict(lambda:{}) # Normalized DCG
##        self.precision=defaultdict(lambda:{}) # Exclusively precision score. 1 if right, 0 otherwise

        self.output_filename=dump_filename

        # stores every kind of score !TODO
        self.scores=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
        self.num_data_points=defaultdict(lambda:defaultdict(lambda:0))

        # old list to store results
        self.overall_results=[]

        # new results in Pandas dataframe
##        self.all_results=DataFrame()

        self.text_results=[]
        self.total_citations=0
        self.numchunks=0
        self.report_file=None
        if results_file:
            self.report_file=codecs.open(corpora.Corpus.dir_output+"report.txt","w")

##        self.citations_extra_info=["a94-1008-cit14","a94-1008-cit4", "a00-1014-cit27"]
        self.citations_extra_info=[]
        self.full_citation_id=""
        self.run_parameters=defaultdict(lambda:None)
        self.csv_columns="file_guid citation_id doc_position query_method doc_method mrr_score rank precision_score az cfc match_guid first_result".split()
        self.dump_straight_to_disk=dump_straight_to_disk
        if dump_straight_to_disk:
            self.startResultsDump()

    def startResultsDump(self):
        """
            Creates a dump .csv file, writes the first line
        """
        filename=getSafeFilename(self.output_filename)
        self.dump_file=codecs.open(filename,"w","utf-8", errors="replace")

        line=u"".join([c+u"\t" for c in self.csv_columns])
        line=line.strip(u"\t")
        line+=u"\n"
        self.dump_file.write(line)

    def dumpResultRow(self,row):
        """
            Writes a single result line directly to the file
        """
        line=""
        try:
            for column_name in self.csv_columns:
                line+=str(row[column_name])+"\t"
            line=line.strip("\t")
            line+="\n"
            self.dump_file.write(line)
        except:
##            print "error writing: ", data_point, sys.exc_info()
            print "error writing: ", sys.exc_info()[:2]


    def computeAverageScores(self):
        """
            Computes the average accuracy of a query_method + doc_method
            combination to later choose the highest one
        """
        self.averages=defaultdict(lambda:defaultdict(lambda:{}))
        self.mrr_averages=defaultdict(lambda:{})

        for metric in self.scores:
            for qmethod in self.scores[metric]:
                for doc_method in self.scores[metric][qmethod]:
                    data_points=self.scores[metric][qmethod][doc_method]
                    self.averages[qmethod][doc_method][metric]=data_points/float(self.num_data_points[qmethod][doc_method])

    def showFinalSummary(self, overlap_in_methods=None):
        """
            Print a nice summary of everything
        """
        print
        print "========================================================================="
        print "Docs processed:", self.numitems
        print "Total citations:", self.total_citations
        print "Chunks:",self.numchunks
        now2=datetime.datetime.now()-self.start_time
        print "Total execution time",now2
        print
        print "Similarity:",self.run_parameters["similarity"]

        overall_precision=defaultdict(lambda:0)
        total_points=defaultdict(lambda:0)

        for qmethod in self.scores["precision"]:
            for method in self.scores["precision"][qmethod]:
                overall_precision[method]+=self.scores["precision"][qmethod][method]/float(self.total_citations)
                total_points[method]+=1

        for method in overall_precision:
            print "Precision for", method, "%02.4f"%(overall_precision[method]/float(total_points[method]))

        print "Saved to filename:", self.output_filename

        if overlap_in_methods:
            self.computeOverlap("rank",overlap_in_methods)

    def writeDataToCSV(self, filename=None):
        """
            Export a CSV with all the data
        """
        if not filename:
            filename=self.output_filename
##        self.all_results.to_csv(filename)
        writeDictToCSV(self.csv_columns, self.overall_results,filename)
        self.output_filename=filename

    def logReport(self,what):
        """
            Throw anything at this, its representation will be written in the report file
        """
        if self.report_file:
            if isinstance(what,basestring):
                try:
                    self.report_file.write(what+"\n")
                except:
                    self.report_file.write(what.__repr__()+"\n")
            else:
                self.report_file.write(what.__repr__()+"\n")

    def addQMethod(self,qmethod):
        """
            A method that just deals with creating a dict for each Qmethod. Should really be using defaultdicts
        """
        pass

    def formatReferences(self,retrieved_references,multi,position):
        res="<ol>"
        for index,reference in enumerate(retrieved_references):
            if index < multi:
                css_class="scoring_region"
            elif position==index:
                css_class="original"
            else:
                css_class="wrong"

            text=formatReferenceAPA(reference)
            res+="<li class="+css_class+">"+text+"</li>"
        res+="</ol>"

    def measureScoreAndLog(self, retrieved_docs, citation_multi, result_dict):
        """
            file_guid, retrieved_docs, right_citation, doc_position, qmethod, method, query, az, cfc
            guid,retrieved,m,doc_position,qmethod,method, queries[qmethod]["text"], az, cfc

            Computes all scores and position document was ranked in
        """
        # docs must be already ordered in descending order
        guids=[]
        for doc in retrieved_docs:
            guid=doc[1]["guid"]
            if guid not in guids:
                guids.append(guid)

        mrr_score=ndcg_score=precision_score=0
        found_at_index=None
        citation_multi=1
        rank=0

        for index, guid in enumerate(guids):
            rank=None
            found_at_index=index
            if guid==result_dict["match_guid"]:
                if citation_multi and citation_multi > 1:
                    if index+1 <= citation_multi:
                        ndcg_score=1
                        precision_score=1
                        mrr_score=1
                        rank=1
                        break
                    else:
                        ndcg_score=1/ math.log(index+2)
                        precision_score=0
                        mrr_score=1/ float(index+1)
                        rank=index+1
                        break
                else:
                    precision_score=1 if index==0 else 0
                    ndcg_score=1/ math.log(index+2)
                    mrr_score=1/ float(index+1)
                    rank=index+1
                    break

        if rank is None:
            rank=-1
            precision_score=0
            mrr_score=0
            ndcg_score=0

        self.logReport("Rank: "+str(found_at_index))
        self.logReport("Correct: "+result_dict["match_guid"]+" Retrieved: "+guids[0])
        result_dict["mrr_score"]=mrr_score
        result_dict["precision_score"]=precision_score
        result_dict["ndcg_score"]=ndcg_score
        result_dict["rank"]=rank
        result_dict["first_result"]=guids[0]
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
            # !TODO Proper change this to pandas
            try:
                self.overall_results.append(result_dict)
            except:
                print "OUT OF MEMORY ERROR!"
                print "Dumping results so far to file..."
                if self.output_filename:
                    self.scores=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
                    self.writeDataToCSV()
                    self.overall_results=[]

##        self.all_results=self.all_results.append(DataFrame(result_dict))

        qmethod=result_dict["query_method"]
        method=result_dict["doc_method"]

        # these aren't necessary: it could be rebuilt from the overall_results at the end, but it's faster
        self.scores["mrr"][qmethod][method]+=result_dict["mrr_score"]
        self.scores["ndcg"][qmethod][method]+=result_dict["ndcg_score"]
        self.scores["precision"][qmethod][method]+=result_dict["precision_score"]
        self.scores["rank"][qmethod][method]+=result_dict["rank"]

        # this is to compute averages
        self.num_data_points[qmethod][method]+=1

    def computeOverlap(self, overlap_in, overlap_between):
        """
            Shows the overlap between methods
        """
        data=DataFrame(self.overall_results)
        group=data.groupby(["file_guid","citation_id"])

        all_overlaps=[]
        for a,b in group:
            numitems=b.shape[0]
            results=[]
            for i in range(numitems):
                doc_method=b.iloc[i]["doc_method"]
                rank=b.iloc[i][overlap_in]
                if doc_method in overlap_between:
                    results.append(rank)

            this_one=1
            for index in range(len(results)-1):
                if results[index] != results[index+1]:
                    this_one=0
                    break

            all_overlaps.append(this_one)

        print "Overlap between", overlap_between,": %02.4f" % (sum(all_overlaps) / float(len(all_overlaps)))

class resultsLoggerCompare(resultsLogger):
    """
        Same as resultsLogger but compares the overlap between methods
        TODO everything, add to class above
    """
    def __init__(self, results_file=True, to_compare=[]):
        """
            takes a to_compare parameter: list of methods to compare
        """
        resultsLogger.__init__(self, results_file)

        self.methods_overlap=0
        self.total_overlap_points=0
##        self.precision_per_method=defaultdict(lambda:[])

    def addResolutionResultDict(self, result_dict):
        pass

def main():
    pass

if __name__ == '__main__':
    main()
