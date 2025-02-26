#! /bin/bash

pickle_file=BOUSSOLE_sensitivity.pickle
echo picklefile $pickle_file
sbatch --export=ALL,pickle_file=$pickle_file run_sensitivity_template.sbatch
