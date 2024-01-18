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
data_name = "validation"

# Load datasets
folder = '/home/jovyan/test/datasets/ConceptualCaptions/'

# Read the converted_data
if data_name == "validation":
    infile = folder+"conceptual_captions_instruct_val_clean.json"
elif data_name == "training":
    infile = folder+"conceptual_captions_instruct_train_clean.json"
    
with open(infile, "r") as f:
    instructions = json.load(f)


# Image Inspection (whether it can be opened)
n_reduced = 1000
print(len(instructions))

print(len(instructions[:n_reduced]))



# Save the cleaned data as json
if data_name == "validation":
    outfile = folder+"conceptual_captions_instruct_val_clean_reduced.json"
elif data_name == "training":
    outfile = folder+"conceptual_captions_instruct_train_clean_reduced.json"
   
with open(outfile, "w") as f:
    json.dump(instructions[:n_reduced], f, indent=2)