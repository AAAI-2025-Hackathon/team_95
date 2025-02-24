#!/bin/bash

#SBATCH --job-name="gojo"	                        
#SBATCH --partition=peregrine-gpu                   
#SBATCH --qos=gpu_long			  		            
#SBATCH --nodes=1                 		            
#SBATCH --ntasks=1                		            
#SBATCH --cpus-per-task=10         		            
#SBATCH --mem=90g                  		            
#SBATCH --gres=gpu:a100-sxm4-80gb:1                 
#SBATCH --time=10-0:00:00 				            


module purge
module load python/anaconda
module load cuda/11.8
module load gcc/13.2.0

python main_upd.py --seed 42 --dir_checkpoint /dir_checkpoint --epochs 11 --batch_size 16 --use_augmented True
