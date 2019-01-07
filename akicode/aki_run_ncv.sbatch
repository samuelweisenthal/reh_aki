#!/bin/sh

#SBATCH --partition=preempt --time=1-00:00:00
#SBATCH -N 1
#SBATCH --mem=100gb
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 
#SBATCH -a 1-50
#SBATCH -o /results/log%a.txt

module load anaconda 
python fof2_search.py -rs $SLURM_ARRAY_TASK_ID -skgf 0 -ons 1 -nj 1 -cal3 0 -x X_aki_full_dialE_ex.csv -y  y_aki_full_dialE_ex.csv -g groups_aki_full_dialE_ex.csv -o /results -weight 0 -samp 0
