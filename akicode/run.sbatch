#!/bin/sh

#SBATCH --partition=preempt --time=1-00:00:00
#SBATCH -N 1
#SBATCH --mem=300gb
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

module load anaconda 
python main2.py
