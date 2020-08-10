#!/bin/bash
#SBATCH -o /home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/interaction_network/job_reports/output.LP_Train_2_1500_phi_thin_LR5e04.out # STDOUT
#SBATCH --job-name=LP_Train_thin_phi_LR5e04  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/interaction_network
python train_IN.py configs/train_IN.yaml
python plot_IN.py configs/train_IN.yaml
