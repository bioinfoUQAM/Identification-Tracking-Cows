#!/bin/bash
#SBATCH --account=def-banire
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=20:50:00
#SBATCH --mail-user=marcelo_de_araujo.voncarlos@uqam.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out

unset PYTHONPATH
export PYTHONNOUSERSITE=1

#module load python/3.11.5 cuda/12.2  
module restore my_enviroment

source ~/envs/my_env/bin/activate   

cd /home/vonzin/scratch/SNA_2025_v2

# 🚀 Executa o script de inferência
python run_inference_corridor_tracking.py \
  --input-video /home/vonzin/scratch/SNA_2025_v2/Videos_Thomas_180 \
  --output-folder /home/vonzin/scratch/SNA_2025_v2/Videos_Thomas_180_beste.mp4 \
  --yolo-model /home/vonzin/scratch/SNA_2025_v2/best_yolov8_detection.pt \
  --num-classes 29 \
  --efficientnet-weights /home/vonzin/scratch/SNA_2025_v2/lightning_logs/version_64763588_thomas_29ids_99cc_180_360cameras/checkpoints/best_model.ckpt \
  --class-names-dir /home/vonzin/scratch/SNA_2025_v2/Dataset_Augmented_29_class_v1_split_300_robusto_CERTO/train
  
  
  
# 🚀 Executa o script de inferência
python run_inference_corridor_tracking.py \
  --input-video /home/vonzin/scratch/SNA_2025_v2/Videos_Thomas_360 \
  --output-folder /home/vonzin/scratch/SNA_2025_v2/Videos_Thomas_360_beste.mp4 \
  --yolo-model /home/vonzin/scratch/SNA_2025_v2/best_yolov8_detection.pt \
  --num-classes 29 \
  --efficientnet-weights /home/vonzin/scratch/SNA_2025_v2/lightning_logs/version_64763588_thomas_29ids_99cc_180_360cameras/checkpoints/best_model.ckpt \
  --class-names-dir /home/vonzin/scratch/SNA_2025_v2/Dataset_Augmented_29_class_v1_split_300_robusto_CERTO/train
  
  
  
