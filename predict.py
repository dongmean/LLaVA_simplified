import os
import json
import torch

from PIL import Image

from model.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils import *
from model.builder import load_pretrained_model

import argparse


def test(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # load eval_data
    data_path = "/home/jovyan/test/datasets/ConceptualCaptions/conceptual_captions_instruct_val_clean.json"
    image_folder = "/home/jovyan/test/datasets/ConceptualCaptions/val/"
    list_data_dict = json.load(open(data_path, "r"))
    
    prompt = "<s>[INST] " + DEFAULT_IMAGE_TOKEN + '\n' + " Generate a caption of this image: [/INST]"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Generate responses
    trial = 3
    results = []
    for i, data in enumerate(list_data_dict):
        for j in range(trial):
            result = {}
            image_file = data['image'].split('/')[-1]

            image = Image.open(image_folder+image_file).convert('RGB')           
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=64,
                    use_cache=True,)
                
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            #print("\n\n*** {}-th output ***".format(i))
            #print("image_file: ", image_file)
            #print("caption: ", outputs)
            
            result['index'] = str(i+1)
            result['image'] = data['image']
            result['instruction'] = data['instruction']
            result['gt_caption'] = data['response']
            result['pred_output'] = outputs
            
            results.append(result)
        
        if i>99:
            break
    
    outfile = "predictions.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    
    test(args)