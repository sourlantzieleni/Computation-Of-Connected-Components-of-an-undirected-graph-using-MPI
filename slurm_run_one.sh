#!/bin/bash
#SBATCH --job-name=MPI
#SBATCH --partition=rome
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=6
#SBATCH --output=my_job.stdout
#SBATCH --error=my_job.stderr
#SBATCH --time=3:00
#SBATCH --mem=64000

GRAPH_FILE="/path/to/file"

module purge
unset UCX_ROOT
unset UCX_NET_DEVICES
unset UCX_TLS
unset OMPI_MCA_btl
unset OMPI_MCA_pml

module load gcc/9.2.0 openmpi/3.1.4

mpicc -O3 -g colouringCC.c mmio.c converter.c -o parallel -lm

srun ./parallel "$GRAPH_FILE"
