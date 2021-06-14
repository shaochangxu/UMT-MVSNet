#!/bin/bash
data=$(date +"%m%d")
n=4
batch=8
epochs=2
d=8
interval_scale=1.06
lr=0.001
lr_scheduler=cosinedecay
loss=unsup_loss
optimizer=Adam
loss_w=4
image_scale=0.25
view_num=7

name=${date}_checkpoints
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/hadoop/scx/mvsnet/anaconda3/envs/drmvsnet/bin/python -m torch.distributed.launch --nproc_per_node=$n --master_port 10190 train.py  \
        --dataset=dtu_yao \
        --batch_size=${batch} \
        --trainpath="/home/hadoop/scx/mvsnet/trainingdata/dtu_training" \
        --loss=${loss} \
        --lr=${lr} \
        --epochs=${epochs} \
        --loss_w=$loss_w \
        --lr_scheduler=$lr_scheduler \
        --optimizer=$optimizer \
        --view_num=$view_num \
        --image_scale=$image_scale \
        --using_apex \
        --reg_loss=True \
        --ngpu=${n} \
        --trainlist=lists/dtu/train.txt \
        --vallist=lists/dtu/val.txt \
        --testlist=lists/dtu/test.txt \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./logdir/${name} \
        --savedir = ./checkpoints \
        2>&1|tee ./${name}-${now}.log &
