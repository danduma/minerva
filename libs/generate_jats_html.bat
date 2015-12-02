rem @echo off
set tmp_file="g:\nlp\phd\pmc\JATSDTD\temp.tmp"
copy %1 %tmp_file%
java -cp "saxon9he.jar;xml-resolver-1.2.jar" net.sf.saxon.Transform -s:%tmp_file% -xsl:JATSPreviewStylesheets\xslt\main\jats-html.xsl -o:%2 -catalog:JATSDTD\catalog-jats-v1-1d3-no-base.xml
del /q %tmp_file%