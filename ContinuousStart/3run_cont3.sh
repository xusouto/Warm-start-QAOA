#!/bin/bash
#SBATCH -J cont3
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=304               # 64 total tasks (processes)
#SBATCH --cpus-per-task=1         # 1 hw-thread per task
#SBATCH --mem-per-cpu=1G          # 64 * 3G = 192G total (ensure node has this)

module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators

for i in $(seq 1 3 58); do  
  for j in $(seq 1 6 97); do 
    srun --exclusive -N1 -n1 --cpu-bind=threads \
         python cont3.py --time "${i}" --steps "${j}" &
  done
done

wait

conda deactivate

