#!/usr/bin/env bash

# SCENE=$1

SWITCH_NERF_PATH=/data/user/Projects/Switch-NeRF
DATASET_PATH=/ssd/user
CKPT_PATH=$DATASET_PATH/checkpoints/switch-nerf

mkdir $DATASET_PATH/exp_switch-nerf

cd $SWITCH_NERF_PATH

SCENES=("building" "campus" "residence" "rubble" "sci-art")

for((i=0;i<${#SCENES[@]};i++));
do
    EXP_PATH=$DATASET_PATH/exp_switch-nerf/${SCENES[i]}
    mkdir $EXP_PATH

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
        --use_env --master_port=12345 --nproc_per_node=4 -m \
        switch_nerf.eval_image \
        --config=switch_nerf/configs/switch_nerf/${SCENES[i]}.yaml \
        --use_moe --exp_name=$EXP_PATH \
        --dataset_path=$DATASET_PATH/${SCENES[i]} \
        --i_print=1000 \
        --moe_expert_type=seqexperts \
        --model_chunk_size=131072 \
        --ckpt_path=$CKPT_PATH/${SCENES[i]}.pt \
        --expertmlp2seqexperts \
        --use_moe_external_gate \
        --use_gate_input_norm

done
