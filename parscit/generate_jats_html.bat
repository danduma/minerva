@echo off
copy %1 
del /q temp.tmp

java  -jar saxon9he.jar %1 JATSPreviewStylesheets\xslt\main\jats-html.xsl -o %2