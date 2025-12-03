#!/bin/bash
#SBATCH -J cont2
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=250              # 64 total tasks (processes)
#SBATCH --cpus-per-task=1         # 1 hw-thread per task
#SBATCH --mem-per-cpu=200M          # 64 * 3G = 192G total (ensure node has this)

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators

for i in $(seq 0 249); do
  srun --exclusive -N1 -n1 --cpu-bind=threads \
       python cont2.py --iter "${i}" &
done

wait

conda deactivate
