import os

import torch
import torchvision
from torchvision import transforms

from cneuro_data_s import algonauts_dataset

import logging
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

root_data_dir = '/engram/nklab/datasets/'

class Uint8ToFloat(object):
    """
    Custom transform that converts a torch uint8 tensor to a float tensor
    scaled to [0, 1]. If the input is not a uint8 tensor, it returns the input unchanged.
    """

    def __call__(self, x):
        # Check if x is a tensor and its type is uint8
        if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        return x

class args_explore():
    def __init__(self):
        self.image_size = 224
        self.data_dir = '/engram/nklab/hossein/recurrent_models/algonauts2025/'
        self.subj = 1
        self.batch_size = 2
        self.backbone_arch = 'dinov2_q'
        self.modality = 'visual audio text'
        self.distributed = 0
        self.num_workers = 0
        self.val_split = 's06e'
        self.num_frames = 20
        self.objective = None
        self.text_bb = 'bert'
        self.video_bb = 'timesformer'
        self.readout_res = 'voxels'  # 'parcels' or 'voxels' or 'hemis'

def transform_img():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform_img = transforms.Compose([
        Uint8ToFloat(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform_img

if __name__ == "__main__":
    args = args_explore()

    print("Loading datasets...")
    train_dataset = algonauts_dataset(
        args, transform=transform_img(), exclude_split=args.val_split, 
    )

    val_dataset = algonauts_dataset(
        args, transform=transform_img(), include_split=args.val_split,
    )

    print(f"Number of train datapoints: {len(train_dataset)}") # 130656
    print(f"Number of validation datapoints: {len(val_dataset)}") # 22985

    print("Inspecting a sample from the validation dataset...")
    media, fmri = val_dataset[0]
    if args.readout_res == "voxels":
        args.num_voxels = len(fmri["sub_1"])
        args.parcellation = val_dataset.parcellation
        args.masked_parcellation = val_dataset.masked_parcellation
    else:
        args.parcellation = None
        args.masked_parcellation = None

    print(media.keys()) # dict_keys(['split', 'ind', 'text', 'text_clip', 'text_llama', 'visual', 'audio', 'sr'])
    print(fmri.keys()) # dict_keys(['sub_1', 'sub_2', 'sub_3', 'sub_5'])

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, args.batch_size, 
        drop_last=False, num_workers=args.num_workers
    ) 
    
    iteration = iter(val_dataloader)
    for i in range(30):  
        _, _ = next(iteration)  
    media, fmri_data = next(iteration)
    
    print(fmri_data['sub_1'].shape)

    print(media['text']['input_ids'].shape) # torch.Size([2, 1, 128])  
    print(media['text']['token_type_ids'].shape) # torch.Size([2, 1, 128])  
    print(media['text']['attention_mask'].shape) # torch.Size([2, 1, 128])  
    print(media['text_clip'].shape) # torch.Size([2, 1, 77])
    print(media['text_llama']['input_ids'].shape) # torch.Size([2, 1, 128]) 
    print(media['text_llama']['attention_mask'].shape) # torch.Size([2, 1, 128]) 
    print(media['visual'].shape) # raw: torch.Size([2, 20, 3, 224, 336])  
    print(media['audio'].shape)  # torch.Size([2, 357601])  

    