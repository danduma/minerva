Minerva
=================

Minerva is an open framework for context-based citation recommendation experiments.

If you want to run an experiment in natural language processing or information retrieval using a corpus of scientific papers you may need to do one, more or all of these things:

- Annotate citations in the running text
- Parse text from the References/Bibliography section
- Find the document for a particular reference inside your corpus
- Split sentences in the text (often less trivial than you expect)
- Deal with the XML schema of the corpus
- Extract position-relevant text from the document:
	- Select sentences around a citation token
	- The full paragraph containing a reference to a Figure
	- The Abstract
	- All sentences in which a particular reference is cited

If you are unlucky and need to use a corpus that was not already converted to a machine-readable representation (e.g. XML), you may also need to:
- Fetch/download a number of files (normally PDFs)
- Convert these PDF files into a structured representation
- Clean up the output from this

Minerva aims to make all of this as easy as possible, by providing built-in solutions for many of these tasks and wrappers around existing tools that deal with many other tasks.

Installing Minerva
=======================

to be continued...


