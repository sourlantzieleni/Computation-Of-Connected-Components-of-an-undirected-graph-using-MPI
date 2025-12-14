#!/bin/bash
#SBATCH --job-name=MPI
#SBATCH --partition=rome
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4         # allocate for the maximum ntasks-per-node
#SBATCH --cpus-per-task=6
#SBATCH --output=my_job.stdout
#SBATCH --error=my_job.stderr
#SBATCH --time=03:00
#SBATCH --mem=60000

GRAPH_FILE="/path/to/file"

module purge
unset UCX_ROOT
unset UCX_NET_DEVICES
unset UCX_TLS
unset OMPI_MCA_btl
unset OMPI_MCA_pml

module load gcc/9.2.0 openmpi/3.1.4

# Compile once
mpicc -O3 -g colouringCC.c mmio.c converter.c -o parallel -lm

# One extra iteration with a single node and a single task
nodes=1
ntasks_per_node=1
total_tasks=$(( nodes * ntasks_per_node ))

echo "Running single-node single-task iteration: nodes=${nodes}, ntasks-per-node=${ntasks_per_node} (total tasks=${total_tasks})"
srun --nodes=${nodes} --ntasks-per-node=${ntasks_per_node} --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --output="run_n${nodes}_tpn${ntasks_per_node}.out" \
     --error="run_n${nodes}_tpn${ntasks_per_node}.err" \
     ./parallel "${GRAPH_FILE}"

# Loop over the requested ntasks-per-node values (on two-node allocation)
for ntasks_per_node in 1 2 3 4; do
  nodes=2
  total_tasks=$(( nodes * ntasks_per_node ))

  echo "Running with nodes=${nodes}, ntasks-per-node=${ntasks_per_node} (total tasks=${total_tasks})"
  # Run and write per-run stdout/stderr files to keep outputs separate
  srun --nodes=${nodes} --ntasks-per-node=${ntasks_per_node} --cpus-per-task=${SLURM_CPUS_PER_TASK} \
       --output="run_n${nodes}_tpn${ntasks_per_node}.out" \
       --error="run_n${nodes}_tpn${ntasks_per_node}.err" \
       ./parallel "${GRAPH_FILE}"
done



echo "All runs finished."
