#!/bin/bash
#SBATCH -J qiskit_64_fixedout
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1              # 64 total tasks (processes)
#SBATCH --cpus-per-task=1         # 1 hw-thread per task
#SBATCH --mem-per-cpu=200M          # 64 * 3G = 192G total (ensure node has this)


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


t=$1  
n=$1

srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads python round2_graph.py --n "$1" --flag "$2" --iter "$3" --folder RGW&
wait
echo "Graphs generated:    $(date)"
# Run while t > n/2
while [ "$t" -gt $(( n / 2 )) ]; do
  # GW pre-solver 
  srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
    python round2_cuts.py --n "$t" --flag "$2" --iter "$3" --folder RGW
  echo "Cuts generated:    $(date)"
  CUTS_JSON="RGW/Cuts/cuts${t}_$2_$3.json"
  NUMCUTS=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("numcuts",0)))' "$CUTS_JSON")

  # Correlation + reduction
  srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
    python round2_corr_rgw.py --n "$t" --flag "$2" --iter "$3" 
  echo "Correlation matrix built:    $(date)"

  t=$((t-1))
done

# Final reconstruction
srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
  python round2_fin.py --n "$1" --t "$t" --flag "$2" --iter "$3" --folder RGW

# Wait for all tasks to complete
wait

conda deactivate

echo "========================================="
echo "JOB END"
echo "Job ID:     $SLURM_JOB_ID"
echo "End time:   $(date)"
echo "========================================="