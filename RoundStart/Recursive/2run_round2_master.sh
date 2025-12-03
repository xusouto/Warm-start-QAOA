
for n in 20; do
  for j in True; do
    for i in $(seq 0 99); do
      sbatch 2run_round2_gw.sh $n $j $i
    done
  done
done
