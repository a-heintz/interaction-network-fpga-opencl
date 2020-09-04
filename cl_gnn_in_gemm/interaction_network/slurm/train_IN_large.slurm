#!/bin/bash
#SBATCH -o /home/aheintz/cl_gnn_in_gemm/interaction_network/job_reports/output.train.out # STDOUT
#SBATCH --job-name=fpga-IN # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=ah2263@cornell.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/aheintz/cl_gnn_in_gemm/interaction_network
python train_IN_large.py ./configs/train_IN_LP_0pt5.yaml
python train_IN_large.py ./configs/train_IN_LP_0pt75.yaml

python process_data.py ./configs/train_IN_LP_0pt5.yaml
python process_data.py ./configs/train_IN_LP_0pt75.yaml
