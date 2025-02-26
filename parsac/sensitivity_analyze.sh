#! /bin/bash

pickle_file=BOUSSOLE_sensitivity.pickle
pickle_file_analyze=BOUSSOLE_sensitivity.analyze.cv.pickle
echo picklefile $pickle_file
file_txt=BOUSSOLE_sensitivity_CV.txt
parsac sensitivity analyze $pickle_file --pickle=$pickle_file_analyze cv  > $file_txt
