@echo off
del c:\nlp\minerva_git\*.* /s /q /f
mkdir c:\nlp\minerva_git
xcopy c:\nlp\minerva\.git\*.* c:\nlp\minerva_git /e /y
del c:\nlp\minerva\*.* /s /q /f
xcopy *.* c:\nlp\minerva\ /y
xcopy api\*.* c:\nlp\minerva\api\ /e /y
xcopy az\*.* c:\nlp\minerva\az\ /e /y
xcopy cit_styles\*.* c:\nlp\minerva\cit_styles\ /e /y
xcopy db\*.* c:\nlp\minerva\db\ /e /y
xcopy evaluation\*.* c:\nlp\minerva\evaluation\ /e /y
xcopy evaluation_runs\*.* c:\nlp\minerva\evaluation_runs\ /e /y
xcopy kw_evaluation_runs\*.* c:\nlp\minerva\kw_evaluation_runs\ /e /y
xcopy importing\*.* c:\nlp\minerva\importing\ /e /y
xcopy proc\*.* c:\nlp\minerva\proc\ /e /y
xcopy parscit\*.* c:\nlp\minerva\parscit\ /e /y
xcopy retrieval\*.* c:\nlp\minerva\retrieval\ /e /y
xcopy scidoc\*.* c:\nlp\minerva\scidoc\ /e /y
xcopy scraping\*.* c:\nlp\minerva\scraping\ /e /y
xcopy scripts\*.* c:\nlp\minerva\scripts\ /e /y
xcopy multi\*.* c:\nlp\minerva\multi\ /e /y
xcopy tests\*.* c:\nlp\minerva\tests\ /e /y
xcopy ml\*.* c:\nlp\minerva\ml\ /e /y
xcopy c:\nlp\minerva_git\*.* c:\nlp\minerva\.git\ /e /y
cd c:\nlp\minerva
c:
git add -A .
set commitmessage="fixes"
set /p commitmessage=Commit message (default - "fixes")?:
git commit -m "%commitmessage%"
git push
