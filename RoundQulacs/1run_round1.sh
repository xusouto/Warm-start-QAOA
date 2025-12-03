#!/bin/bash
#SBATCH -J mxc2.1q
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH -p a64
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 64 total tasks (processes)
#SBATCH --cpus-per-task=10         # 1 hw-thread per task
#SBATCH --mem=100M          

module --force purge
source /opt/cesga/Lmod-ft3/setup_qmio_a64.sh
module load qulacs-hpcx/1.0


srun --exclusive -N1 -n1 --cpu-bind=threads python -u round1_qaoa.py "Cuts20/results_g$1.json" --cut "$2"  &

wait

