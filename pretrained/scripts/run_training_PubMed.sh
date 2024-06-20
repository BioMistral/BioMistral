#!/bin/bash
#SBATCH --job-name=BioMistral-PubMed
#SBATCH --constraint=a100
#SBATCH --ntasks=32                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=8          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:8                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=8           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs_slurm/PubMed-LLM%j.out # nom du fichier de sortie
#SBATCH --error=./logs_slurm/PubMed-LLM%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=XXX
#
# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load cpuarch/amd
module load pytorch-gpu/py3/2.0.1

# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u training_mistral_7B-LM.py \
   --model_name="./models/mistralai_Mistral-7B-Instruct-v0.1/" \
   --path_dataset="./datasets/PubMed/" \
   --output_dir="./BioMistral-7B/" \
   --epochs=3 \
   --batch_size=32 \
   --save_steps=80 \
   --logging_steps=10 \
   --seed=42 \
   --learning_rate=2e-05

