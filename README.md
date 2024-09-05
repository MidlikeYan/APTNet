Config
Set paths to datasets and desired log directories in config.py

Train
CUDA_VISIBLE_DEVICES=0 python train_with_print2.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-4 \
    --transform 'imagenet' \
    --lr 0.01 \
    --lr2 0.01 \
    --prompt_size 1 \
    --freq_rep_learn 20 \
    --pretrained_model_path /root/pretrained-models/simgcd_pretrained_model_cub.pt\
    --prompt_type 'patch' \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 1 \
    --model_path work/saved_models
