#!/bin/bash
#SBATCH -J mxc2.1q
#SBATCH -o myjob_%j.o
#SBATCH -e myjob_%j.e
#SBATCH --time=00:10:00
#SBATCH --nodes=5
#SBATCH --ntasks=50               
#SBATCH --cpus-per-task=5       
#SBATCH --mem=1G 


module load qmio/hpc gcc/12.3.0 qiskit-qulacs/0.1.0-python-3.9.9 
module load cvxpy/1.7.1-python-3.9.9

for i in $(seq 0 49); do
    srun --exclusive -N1 -n1 --cpu-bind=threads python -u round2_rqaoa.py --n 20 --flag False --iter "${i}" --case "WSQAOA"&
done

wait
