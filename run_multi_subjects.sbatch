#!/bin/bash

#SBATCH --job-name=rw-msubs
#SBATCH --output=output_test.slurm
#SBATCH -a 31-60
#SBATCH --nodes=1

##SBATCH -a 0-180
##SBATCH --mem-per-cpu=32GB
##SBATCH --cpus-per-task=4
##SBATCH --time=12:00:00
##SBATCH --gres=gpu:2

module purge
module load freesurfer/6.0.0
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load anaconda3/4.3.1
source activate autolabel

export HCP_CREDENTIALS="`cat ~/.hcp-passwd`"

##python multiprocess_labeling.py --sub $SLURM_ARRAY_TASK_ID --hemi lh --scorefunc_type 0
##python multiprocess_labeling.py --sub $SLURM_ARRAY_TASK_ID --hemi rh --scorefunc_type 0
python starting_guess.py --sub $SLURM_ARRAY_TASK_ID --hemi lh --scorefunc_type 0
python starting_guess.py --sub $SLURM_ARRAY_TASK_ID --hemi rh --scorefunc_type 0
