#!/bin/bash
#SBATCH -J qiskit_64_fixedout
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=200M          


module load qmio/hpc miniconda3/22.11.1-1
conda activate simulators


echo "========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "Nodelist:      $SLURM_JOB_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Current dir:   $(pwd)"
echo "Start time:    $(date)"
echo "========================================="


srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads python round2_graph.py --n "$1" --flag "$2" --iter "$3" --folder GW&
wait
echo "Graphs generated:    $(date)"
srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
    python round2_gw_solver.py --n "$1" --flag "$2" --iter "$3"
echo "Cut generated:    $(date)"
wait

conda deactivate

echo "========================================="
echo "JOB END"
echo "Job ID:     $SLURM_JOB_ID"
echo "End time:   $(date)"
echo "========================================="