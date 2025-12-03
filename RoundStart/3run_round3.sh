#!/bin/bash
#SBATCH -J qiskit_64
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=01:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=180            
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=500M        

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators

for i in $(seq 1 6); do
  for j in $(seq 0 29); do
    srun --exclusive -N1 -n1 --cpu-bind=threads python round3.py --depth "${i}" --iter "${j}" &
  done
done

wait

conda deactivate
