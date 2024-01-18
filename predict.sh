export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


python predict.py \
    --model-path /home/jovyan/test/LMMs/llava_lora_val_lr2e4/ \
    --model-base mistralai/Mistral-7B-Instruct-v0.1 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_beams 1 \