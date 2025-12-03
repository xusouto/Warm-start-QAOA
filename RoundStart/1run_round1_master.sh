
for i in $(seq 0 9); do
  for j in $(seq 0 4); do
    sbatch 1run_round1.sh $i $j
  done
done