#!/bin/bash

TASKS=(open_drawer push_buttons slide_block_to_color_target put_money_in_safe put_groceries_in_cupboard)

for TASK in "${TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python eval.py \
        rlbench.tasks=[$TASK] \
        rlbench.task_name='multi' \
        rlbench.demo_path=$PERACT_ROOT/data/val \
        framework.gpu=0 \
        framework.logdir=$PERACT_ROOT/ckpts/ \
        framework.start_seed=0 \
        framework.eval_envs=1 \
        framework.eval_from_eps_number=0 \
        framework.eval_episodes=10 \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        framework.eval_type='last' \
        rlbench.headless=True
done
