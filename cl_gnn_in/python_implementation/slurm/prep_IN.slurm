#!/bin/bash
#SBATCH --job-name=LP_prep_5     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/FPGA_dev/interaction_network
python prep_LP.py configs/prep_LP.yaml
