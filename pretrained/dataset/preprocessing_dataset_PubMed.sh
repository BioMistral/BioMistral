#!/bin/bash
#SBATCH --job-name=PubMed_prep
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs_slurm/llm_test%j.out # nom du fichier de sortie
#SBATCH --error=./logs_slurm/llm_test%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=XXX
#
# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/2.0.1

# Echo des commandes lancees
set -x -e

# Chargement des modules
export OMP_NUM_THREADS=10

export TMPDIR=$JOBSCRATCH

# From scratch max seq length
srun -l python -u preprocessing_dataset.py \
   --model_name="./models/mistralai_Mistral-7B-Instruct-v0.1" \
   --dataset="./datasets_source/PubMed" \
   --output_dataset_path="./datasets/PubMed/" \
   --batch_size=500000 \
   --seed=42 \
   --preprocessing_num_workers=10
