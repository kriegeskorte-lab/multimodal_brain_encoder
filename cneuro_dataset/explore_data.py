import os
import random

import torch
import torchvision
from torchvision import transforms

import sys
sys.path.append("/engram/nklab/pf2477")
from multimodal_encoder.cneuro_dataset.cneuro_data import algonauts_dataset

import logging
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

root_data_dir = '/engram/nklab/datasets/'

BACKBONE_LIST = {
    'text': {
        "llama": "meta-llama/Llama-3.2-1B", 
        "deberta": "microsoft/deberta-v3-large", 
        "metaclip": "facebook/metaclip-2-worldwide-m16"},
    'video': {
        "timesformer": "facebook/timesformer-base-finetuned-k400", 
        "videomae": "OpenGVLab/VideoMAEv2-Base", 
        "dino": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
        "metaclip": "facebook/metaclip-2-worldwide-m16"},
    'audio': {
        "wav2vec2": "facebook/wav2vec2-base-960h", 
        "whisper": "openai/whisper-small"}
}

class args_explore():
    def __init__(self):
        self.image_size = 224
        self.data_dir = '/engram/nklab/hossein/recurrent_models/algonauts2025/'
        self.subj = 1
        self.batch_size = 2
        self.modality = ['video', 'audio', 'text']
        self.num_workers = 0
        self.num_frames = 20
        self.text_backbone = random.choice(list(BACKBONE_LIST['text'].keys()))
        self.video_backbone = random.choice(list(BACKBONE_LIST['video'].keys()))
        self.audio_backbone = random.choice(list(BACKBONE_LIST['audio'].keys()))
        self.readout_res = 'voxels'  # 'parcels' or 'voxels' or 'hemis'
        self.backbone_list = BACKBONE_LIST

if __name__ == "__main__":
    args = args_explore()

    print("Loading datasets...")
    train_dataset = algonauts_dataset(
        args, include_splits="friends-train-default",
    )
    test_dataset = algonauts_dataset(
        args, include_splits="friends-test-default"
    )

    print(f"Number of train samples: {len(train_dataset)}") # 112649
    print(f"Number of test samples: {len(test_dataset)}") # 22985

    print("Inspecting a sample from the test dataset...")
    media, fmri = test_dataset[0]
    if args.readout_res == "voxels":
        args.num_voxels = len(fmri["sub_1"])
        args.parcellation = test_dataset.parcellation
        args.masked_parcellation = test_dataset.masked_parcellation
    else:
        args.parcellation = None
        args.masked_parcellation = None

    print(media.keys()) # dict_keys(['split', 'ind', 'text', 'video', 'audio']) 
    print(fmri.keys()) # dict_keys(['sub_1'])

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, args.batch_size, 
        drop_last=False, num_workers=args.num_workers
    ) 
    
    iteration = iter(test_dataloader)
    # for i in range(50):  
    #     _, _ = next(iteration)  
    media, fmri_data = next(iteration)
    print(fmri_data['sub_1'].shape) # torch.Size([2, 172218])
    print(media['split']) # ['s01e01a', 's01e01a'] 
    print(media['ind']) # tensor([0, 1])

    if 'text' in media:
        if args.text_backbone in ['llama', 'deberta', 'metaclip']:
            print(args.text_backbone)
            print("text input_ids", media['text']['input_ids'].shape) # torch.Size([2, 128])
            print("text attention_mask", media['text']['attention_mask'].shape) # torch.Size([2, 128])
            if args.text_backbone == 'deberta':
                print("text token_type_ids", media['text']['token_type_ids'].shape) # torch.Size([2, 128])
    
    if 'video' in media:
        print(args.video_backbone)
        print("video pixel_values", media['video']['pixel_values'].shape) # torch.Size([2, 16, 3, 224, 224])
        
    if 'audio' in media:
        print(args.audio_backbone)
        if args.audio_backbone == 'wav2vec2':
            print("audio input_values", media['audio']['input_values'].shape) # torch.Size([2, 492818]) 
        elif args.audio_backbone == 'whisper':
            print("audio input_features", media['audio']['input_features'].shape) # torch.Size([2, 80, 3000])
