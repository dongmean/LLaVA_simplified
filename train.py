import os
import copy
from dataclasses import dataclass, field
import json

import pathlib
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset

from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoConfig
from clip_encoder import CLIPVisionTower
from model.llava import LlavaLlamaForCausalLM

from custom_trainer import CustomTrainer

from model.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # Default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None,metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None,metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    train_image_folder: Optional[str] = field(default=None)
    val_image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    report_to: str = "wandb",  # enable logging to W&B
    run_name: str = "none",  # name of the W&B run (optional)
    logging_steps: int = field(default=1),  # how often to log to W&B
    do_eval: bool = field(default=True),
    evaluation_strategy: str = "steps",
    eval_steps: int = field(default=10),


# All layers
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            #print("names:", names)
            
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) 

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# TODO: conversation style... vicuna! 

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    batch = []
    for source in sources:
        q = DEFAULT_IMAGE_TOKEN + '\n' + source['instruction']
        
        a = source['response']
        a = a.strip()
        
        batch.append([q,a])
    #print("***batch: ", batch) #[['<image>\nGenerate a caption of this image: ', 'young adult book by novelist .']
    
    return batch

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources: # [q, a]
        assert len(source) == 2
        
        conversation = '<s>[INST] ' + source[0] +' [/INST]' + source[1] + ' </s>' # Mistral
        
        conversations.append(conversation)
    #print("conversations: ", conversations) #['<image>\nGenerate a caption of this image: like birds on a wire']
    
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    
    # masking instructions
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(SupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        print("***Dataset loaded successfully")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        if 'train' in data_path:
            self.mode = 'train'
            self.image_folder = self.data_args.train_image_folder
        elif 'val' in data_path:
            self.mode = 'val'
            self.image_folder = self.data_args.val_image_folder
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image'].split('/')[-1]
            
            #image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            image = Image.open(self.image_folder+image_file).convert('RGB')           
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
            sources = preprocess_multimodal( copy.deepcopy([e for e in sources]), self.data_args)
        else:
            print("***Warning! No image file in the instruction sample!!")
            sources = copy.deepcopy([e for e in sources])
            
        # construct data_dict
        data_dict = preprocess(sources, self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            data_dict['image'] = image
        
        return data_dict
    
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        #print("input_ids: ", input_ids)
        #print("labels: ", labels)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.train_data_path,
                                data_args=data_args)
    
    eval_dataset = SupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.eval_data_path,
                                data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train():

    # parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # initialize LLM & tokenizer
    model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=training_args.model_max_length, 
                                              padding_side="right", padding='max_length')
    tokenizer.pad_token = tokenizer.unk_token # = 0
  
    # freeze backbones
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        #vision_encoder.vision_tower.requires_grad_(False)
    
    # Specify lora layers
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
                
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    # initialize CLIP_encoder
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.get_vision_tower().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    data_args.image_processor = model.get_vision_tower().image_processor
    print("***Finished loading Models")
    
    # create data_loader
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # fine-tuning
    trainer = CustomTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    trainer.train()
    trainer.save_state()
    
    # save the model
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    
    # Prediction
    
    # Hyperparameters for prediction
    temperature = 0.8
    top_p = 0.95
    num_beams = 1
    
    
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
            image_tensor = data_args.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device),
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
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
    train()