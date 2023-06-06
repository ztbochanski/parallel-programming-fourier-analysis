#!/bin/bash
mpic++ -openmp fourier.cpp -o fourier -lm -fopenmp
mpiexec -mca btl self,tcp -np $1 ./fourier
