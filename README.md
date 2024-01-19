# LLaVA Simplified

# Brief Summary
- This is a simplified tutorial of [LLaVA](https://github.com/haotian-liu/LLaVA), aligning [Mistral-instruct-7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) with [CLIP-ViT-B](openai/clip-vit-base-patch16).
- We fine-tune LLaVA on [ConceptualCaption](https://ai.google.com/research/ConceptualCaptions/download) dataset with 3.3M image-caption pairs (i.e., it is tailored for the captioning task)
- Hope this can facilitate understanding of the [LLaVA](https://github.com/haotian-liu/LLaVA) codebase, which became a bit heavy and verbose recently due to their huge advance.


# How to run

### Fine-tuning
```bash 
deepspeed train.py \
    --deepspeed ds_configs/config_zero2.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --train_data_path /your/data/path/ConceptualCaptions/conceptual_captions_instruct_train_clean.json \
    --eval_data_path /your/data/path/ConceptualCaptions/conceptual_captions_instruct_val_clean.json \
    --train_image_folder /your/data/path/ConceptualCaptions/train/ \
    --val_image_folder /your/data/path/ConceptualCaptions/val/ \
    --vision_tower openai/clip-vit-base-patch16 \
```
Detailed configuration can be found in [`finetune.sh`](https://github.com/dongmean/LLaVA_simplified/finetune.sh)

### Prediction
```bash 
python predict.py \
    --model-path /home/jovyan/test/LMMs/llava_lora_val_lr2e4/ \
    --model-base mistralai/Mistral-7B-Instruct-v0.1 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_beams 1 \
```
More detailed script can be found in [`predict.sh`](https://github.com/dongmean/LLaVA_simplified/predict.sh)

### Data preprocessing
- Data download: Retrieve each image from URL [`data_preprocessing/data_cleaning.py`](https://github.com/dongmean/LLaVA_simplified/data_preprocessing/data_download.py)
- Data preprocessing: Reformat image-text to LLaVA input format [`data_preprocessing/data_cleaning.py`](https://github.com/dongmean/LLaVA_simplified/data_preprocessing/data_preprocessing.py)
- Data cleaning: Remove images not able to open by CLIP image processor [`data_preprocessing/data_cleaning.py`](https://github.com/dongmean/LLaVA_simplified/data_preprocessing/data_preprocessing.py)


# Modified files
- All files in the data_preprocess/ folder
- train.py to only remain the major components (e.g., model initialization, data loader, model training)
- Many files in the model/ folder to reduce the verbosity
- **See find_all_linear_names() in train.py for LoRA configuration**
- **See prepare_inputs_labels_for_multimodal() in model/llava.py for patch-level alignment**
