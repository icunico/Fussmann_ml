* Run Model Calibration * 
-file needed: 
BOUSSOLE_calibration.xml        -> configure calibration
TotalCHLA.obs                   -> observation file 
run_calibration_template.sbatch -> run on HPC machines
-run:
sbatch run_calibration_template.sbatch


* Run Model Sensitivity * 
-file needed: 
BOUSSOLE_sensitivity.xml        -> configure sensitivity
sensitivity_sample.sh           -> launcher of the seensitivity sampling phase
run_sensitivity_template.sbatch -> run sensitivity on HPC machines
sensitivity_run.sh              -> launcher of the sensitivity run phase

-run:
bash sensitivity_sample.sh      -> sampling step
bash sensitivity_run.sh         -> ensemble model run step
bash sensitivity_analyze.sh     -> analysis step

->results are stored in BOUSSOLE_sensitivity_CV.txt


