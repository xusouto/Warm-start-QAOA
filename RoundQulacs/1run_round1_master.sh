
for i in $(seq 0 0); do
  for j in $(seq 0 0); do
    sbatch 1run_round1.sh $i $j
  done
done