# Possible inputs:
# -------------------------------------------------------------------------------------
# sbatch 2run_round2_warm.sh $n $j $i        ---------> Warm-start recursive QAOA
# sbatch 2run_round2_cold.sh $n $j $i        ---------> Cold-start recursive QAOA
# sbatch 2run_round2_gw.sh $n $j $i          ---------> Goemams-Williamson
# sbatch 2run_round2_rgw.sh $n $j $i         ---------> Recursive Goemams-Williamson
# -------------------------------------------------------------------------------------

# Loop:
# -------------------------------------------------------------------------------------
max_in_batch=40                                                                    # Max jobs to submit per batch
batch_count=0                                                                      # Counter for the jobs in the current batch
batch_ids=()                                                                       # Array to store the job IDs

for n in 20; do                                                                    # Number of nodes
  for j in True; do                                                                # Flag (True/False)
    for i in $(seq 0 0); do   # Iteration (0 to 99)
      
      out=$(sbatch 2run_round2_warm.sh "$n" "$j" "$i")
      echo "$out"
      jobid=$(echo "$out" | awk '{print $4}')
      batch_ids+=("$jobid")
      batch_count=$((batch_count + 1))

      # Wait for finish
      if [ "$batch_count" -eq "$max_in_batch" ]; then
        echo "Submitted $max_in_batch jobs, waiting for them to finish..."
        
        # Wait for all jobs in the batch to complete
        while :; do
          still_running=0
          for id in "${batch_ids[@]}"; do
            # Check if the job is still running using squeue
            if squeue -h -j "$id" > /dev/null 2>&1; then
              still_running=1
              break
            fi
          done

          # If no jobs are running, we know the batch is finished
          if [ "$still_running" -eq 0 ]; then
            echo "Batch finished."
            break
          fi

          sleep 30   # Avoid overwhelming the scheduler with constant checks
        done

        # Reset the batch
        batch_ids=()
        batch_count=0
      fi
    done
  done
done

# Handle any remaining jobs (if total jobs are not a multiple of 40)
if [ "$batch_count" -gt 0 ]; then
  echo "Waiting for final $batch_count jobs to finish..."
  
  while :; do
    still_running=0
    for id in "${batch_ids[@]}"; do
      # Check if the job is still running using squeue
      if squeue -h -j "$id" > /dev/null 2>&1; then
        still_running=1
        break
      fi
    done

    # If no jobs are running, batch is complete
    if [ "$still_running" -eq 0 ]; then
      echo "Final batch finished."
      break
    fi

    sleep 30   # Avoid constant polling
  done
fi
# -------------------------------------------------------------------------------------
