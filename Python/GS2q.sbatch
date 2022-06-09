#!/bin/bash 
#how long the job will run
#SBATCH --time=10:00:00
# name the output file where data will go
#SBATCH -o GS.out
#which queue/nodes will the job go to
#SBATCH -p nodes
#number of nodes and cores to be used
#SBATCH -N 1
#SBATCH -n 10

# The following must be done in the terminal in order to create the virtual environment.
# It may work here in the sbatch file, we are not sure.
    # virtualenv -p pyhton3 .virtualenvs/py2
    #                        ^^^^^^^^^^^^^^^ This is the directory I chose for the virtual
    #                                        environment. It can be any directory.

# Direct sbatch to use the virtual environment
    source .virtualenvs/py2/bin/activate

# The following modules needed to be installed in the virtual environment.
# It can be done in the terminal after using the activate command in the terminal above,
# or it may be able to be done in this sbatch file instead (not sure).
    # pip install --upgrade
    # pip install numpy
    # pip install pillow
    # pip install matplotlib

# call the executable
    python3 Generate_Gray_Scott_Data.py