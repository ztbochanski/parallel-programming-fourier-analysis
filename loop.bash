#!/bin/bash
module load slurm
module load openmpi

NUM_PROCESSORS=(1 2 4 6 8)

for np in "${NUM_PROCESSORS[@]}"
do
    sbatch -J fouri_$np -A cs475-575 -p classmpitest -N $np -n $np --constraint=ib --tasks-per-node=1 -o mpiproj_$np.out -e mpiproj_$np.err submit.bash $np
done
