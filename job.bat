#!/bin/bash
#OAR -l /nodes=1/core=16,walltime=48:00:00
#OAR -n name_job
#OAR --project tamtam

set -e

source /applis/site/env.bash
source /applis/site/guix-start.sh


/home/gomesdaa/.guix-profile/bin/mpirun -np 16 /home/gomesdaa/.guix-profile/bin/lmp_mpi -i code.lammps -log log.lammps












