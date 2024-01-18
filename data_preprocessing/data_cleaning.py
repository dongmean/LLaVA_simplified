import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import json

from PIL import Image
import torch
from transformers import CLIPImageProcessor



# number of processes in the pool can be larger than cores
num_processes = 32
images_per_part = 100

# Data mode
data_name = "training"

# Load datasets
folder = '/home/jovyan/test/datasets/ConceptualCaptions/'

# Read the converted_data
if data_name == "validation":
    infile = folder+"conceptual_captions_instruct_val.json"
elif data_name == "training":
    infile = folder+"conceptual_captions_instruct_train.json"
    
with open(infile, "r") as f:
    instructions = json.load(f)


# CLIP model
print("***start loading CLIP")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Image Inspection (whether it can be opened)
instructions_clean = []
error_cnt = 0
for data in tqdm(instructions):
    image_file = data['image'].split('/')[-1]

    try:
        image = Image.open(folder+'train/'+image_file).convert('RGB')
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        instructions_clean.append(data)
    except:
        error_cnt+=1
print("# of wrong images: ", error_cnt)



# Save the cleaned data as json
if data_name == "validation":
    outfile = folder+"conceptual_captions_instruct_val_clean.json"
elif data_name == "training":
    outfile = folder+"conceptual_captions_instruct_train_clean.json"
    
with open(outfile, "w") as f:
    json.dump(instructions_clean, f, indent=2)