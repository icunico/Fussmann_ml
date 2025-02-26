#! /bin/bash

pickle_file=BOUSSOLE_sensitivity.pickle
xml_file=BOUSSOLE_sensitivity.xml
echo xmlfile $xml_file
echo picklefile $pickle_file
parsac sensitivity sample $xml_file $pickle_file random 5
