#!/bin/bash
#SBATCH -A research
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-0:00:00
#SBATCH --mail-user=pranav.m@research.iiit.ac.in
#SBATCH --mail-type=ALL



#SBATCH --output=Nerf.txt
set -x

PROJECT_NAME=Nerf
SHARE_ROOT_DIR=solid@ada:/share3/solid/$PROJECT_NAME/
#DATASET_NAME=THumans2_70_views
GNODE_ROOT_DIR=/ssd_scratch/cvit/solid/$PROJECT_NAME/
SRC_CODE_DIR=/home2/solid/CV-Project
#EXP_NAME=rgb_and_cse
#REPRESENTATION_NAME=convole3
#MODEL_NAME=pix2pix_rgb_and_cse
#DATASET_MODE=aligned_rgb_and_cse
#INPUT_NC=6
#PREPROCESS=crop


cd $SRC_CODE_DIR
conda init bash
eval "$(conda shell.bash hook)"
conda activate depref

python train_hash.py --write
