#!/bin/bash
#SBATCH --account=def-banire
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:50:00
#SBATCH --mail-user=marcelo_de_araujo.voncarlos@uqam.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out

unset PYTHONPATH
export PYTHONNOUSERSITE=1

#module load python/3.11.5 cuda/12.2  
module restore my_enviroment

source ~/envs/my_env/bin/activate   

cd /home/vonzin/scratch/SNA_2025_v2


python treinamento_detalhado.py \
  --data-dir /home/vonzin/scratch/SNA_2025_v2/Data_Adicional/thomacrops_augmented \
  --batch-size 32 \
  --lr 0.001 \
  --epochs 100


