#!/bin/bash
#SBATCH -J mxc2.1
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10               
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=500M          

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators


for i in $(seq 0 9); do
  srun --exclusive -N1 -n1 --cpu-bind=threads \
       python round1_cuts.py --graph "${i}"  &
done

wait

conda deactivate

