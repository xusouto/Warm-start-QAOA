#!/bin/bash
#SBATCH -J cont1_p
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1              # 64 total tasks (processes)
#SBATCH --cpus-per-task=1         # 1 hw-thread per task
#SBATCH --mem-per-cpu=1G          # 64 * 3G = 192G total (ensure node has this)

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators


    
srun --exclusive -N1 -n1 --cpu-bind=threads python cont1_plot.py \
  --indir Results1 \
  --out-probs Results1/P_E_vs_depth.png \
  --out-energies Results1/Energies_vs_depth.png &



wait

conda deactivate

