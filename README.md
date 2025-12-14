### Parallel and Distributed Systems Assignment 2
---
This repository is used for the purposes of assignment 2 of PDS class. MPI is used for label propagation in large undirected graphs to find the connected components.
The algorithm was ran on the HPC of Aristotle University of Thessaloniki using slurm. 
The necessary files are provided to facilitate running the code on the HPC.

Run:
```bash
sbatch slurm_run_many.sh
```
to run the algorithm on a graph with various ranks and 
```bash
sbatch slurm_run_one.sh
```
to test only one configuration.

Stderr is redirected to my_job.stderr and stdout to my_job.stdout by default.
Please edit the .sh files to provide the path for the .mtx file and to tweak any slurm parameters you wish.

You could use
```bash
tail -f my_job.stdout
```
if you wish to watch the output in real time.
