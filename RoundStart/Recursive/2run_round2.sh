#!/bin/bash
#SBATCH -J qiskit_64_fixedout
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5              # 64 total tasks (processes)
#SBATCH --cpus-per-task=10         # 1 hw-thread per task
#SBATCH --mem-per-cpu=3G          # 64 * 3G = 192G total (ensure node has this)


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

BASE="/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/RoundStart/Recursive"
t=$1  # Initialize t to n
n=$1

# First task with initial iteration
srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads python round2_graph.py --n "$1" --flag "$2" --iter "$3" &
wait
echo "Graphs generated:    $(date)"
# Main loop: run while t > n/2
while [ "$t" -gt $(( n / 2 )) ]; do
  # GW pre-solver (serial)
  srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
    python round2_gw.py --n "$t" --flag "$2" --iter "$3"
  echo "Cuts generated:    $(date)"
  CUTS_JSON="$BASE/Cuts/cuts${t}_$2_$3.json"
  NUMCUTS=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("numcuts",0)))' "$CUTS_JSON")

  # Loop over the cuts (0..NUMCUTS-1)
  for i in $(seq 0 $((NUMCUTS-1))); do
    srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
      python round2_qaoa.py --n "$t" --flag "$2" --iter "$3" --cut "$i" &
  done
  wait
  echo "QAOAs done:    $(date)"
  # Correlation + reduction (pass numcuts if your script expects it)
  srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
    python round2_corr.py --n "$t" --flag "$2" --iter "$3" --cut "$NUMCUTS"
  echo "Correlation matrix built:    $(date)"
  # Decrement
  t=$((t-1))
done

# Final reconstruction
srun -n1 --cpus-per-task=1 --exclusive --cpu-bind=threads \
  python round2_fin.py --n "$1" --t "$t" --flag "$2" --iter "$3"

# Wait for all tasks to complete
wait

conda deactivate

echo "========================================="
echo "JOB END"
echo "Job ID:     $SLURM_JOB_ID"
echo "End time:   $(date)"
echo "========================================="