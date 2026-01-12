#!/bin/bash
#SBATCH -J mxc2.1q
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 64 total tasks (processes)
#SBATCH --cpus-per-task=1         # 1 hw-thread per task
#SBATCH --mem=200M          

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators


srun --exclusive -N1 -n1 --cpu-bind=threads python round1_plot.py \
  --folder Results1 \
  --pattern "results1_g*_c*.json" \
  --with-error std \
  --png Energy_vs_eps.png \
  --csv Averaged_energy.csv


wait

conda deactivate

