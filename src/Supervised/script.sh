#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=jacob.k.johnson@utah.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -o filename-%j

#echo 0

source /scratch/general/vast/u0403624/miniconda3/etc/profile.d/conda.sh

#echo conda

conda activate 38b

#echo activated

wandb disabled
export TRANSFORMER_CACHE="../../../cache"

#echo setup

bash run_single_unifiedqa.sh unifiedqa-v2-t5-3b-1251000 ../../../out/
bash eval_single_unifiedqa_model.sh unifiedqa-v2-t5-3b-1251000 ../../../out/
bash compute_unifiedqa_stats.sh unifiedqa-v2-t5-3b-1251000 ../../../out/
