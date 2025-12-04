# Possible inputs:
# -------------------------------------------------------------------------------------
# sbatch 2run_round2_warm.sh $n $j $i        ---------> Warm-start recursive QAOA
# sbatch 2run_round2_cold.sh $n $j $i        ---------> Cold-start recursive QAOA
# sbatch 2run_round2_gw.sh $n $j $i          ---------> Goemams-Williamson
# sbatch 2run_round2_rgw.sh $n $j $i         ---------> Recursive Goemams-Williamson
# -------------------------------------------------------------------------------------

# Loop:
# -------------------------------------------------------------------------------------
for n in 20; do                            # ---------> Number of nodes
  for j in True; do                        # ---------> Fully-connected weighted graph (True) or p(E)=1/2 w=1 graph (False)
    for i in $(seq 0 99); do
      sbatch 2run_round2_gw.sh $n $j $i
    done
  done
done
# -------------------------------------------------------------------------------------
