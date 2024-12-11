#!/usr/bin/env bash

export PYTHONDONTWRITEBYTECODE=1

SCENE=$1

DATASET_PATH=/ssd/user
MEGA_NERF_PATH=/data/user/Projects/mega-nerf
CKPT_PATH=$DATASET_PATH/checkpoints/mega-nerf

mkdir $DATASET_PATH/exp_mega-nerf
mkdir $EXP_PATH
cd $MEGA_NERF_PATH

SCENES=("building" "campus" "residence" "rubble" "sci-art")

for((i=0;i<${#SCENES[@]};i++));
do
    EXP_PATH=$DATASET_PATH/exp_mega-nerf/${SCENES[i]}
    mkdir $EXP_PATH

    CUDA_VISIBLE_DEVICES=0 python -m mega_nerf.eval \
        --config_file configs/mega-nerf/${SCENES[i]}.yaml  \
        --exp_name $EXP_PATH \
        --dataset_path $DATASET_PATH/${SCENES[i]} \
        --container_path $CKPT_PATH/${SCENES[i]}-pixsfm-8.pt

done
