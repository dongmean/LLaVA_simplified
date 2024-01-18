export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


deepspeed train.py \
    --deepspeed ds_configs/config_zero2.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --train_data_path /home/jovyan/test/datasets/ConceptualCaptions/conceptual_captions_instruct_train_clean_reduced.json \
    --eval_data_path /home/jovyan/test/datasets/ConceptualCaptions/conceptual_captions_instruct_val_clean_reduced.json \
    --train_image_folder /home/jovyan/test/datasets/ConceptualCaptions/train/ \
    --val_image_folder /home/jovyan/test/datasets/ConceptualCaptions/val/ \
    --vision_tower openai/clip-vit-base-patch16 \
    --mm_vision_select_layer -1 \
    --mm_vision_select_feature patch  \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/jovyan/test/LMMs/llava_lora_train/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU  \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 64 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --report_to wandb \
    --run_name training_validation_loss_mistral_lr2e4 \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 300 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \

    #--evaluation_strategy "steps" \
    #--learning_rate 2e-5 \