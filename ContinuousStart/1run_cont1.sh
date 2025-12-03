#!/bin/bash
#SBATCH -J cont1
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=250              
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=1G          

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators


for i in $(seq 1 5); do
  for j in $(seq 0 49); do
    srun --exclusive -N1 -n1 --cpu-bind=threads \
         python cont1_state.py --depth "${i}" --iter "${j}" &
  done
done

wait

conda deactivate

