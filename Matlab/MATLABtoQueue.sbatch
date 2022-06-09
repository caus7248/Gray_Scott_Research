#!/bin/bash 
#how long the job will run
#SBATCH --time=10:00:00
# name the output file where data will go
#SBATCH -o HF.out
#which queue/nodes will the job go to
#SBATCH -p nodes
#number of nodes and cores to be used
#SBATCH -N 1
#SBATCH -n 1
 
# call your executable (this is where you would call MATLAB to run your script)
/opt/matlab/R2021b/bin/matlab HF_demo