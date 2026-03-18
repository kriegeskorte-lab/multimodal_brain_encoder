import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torchvision.io as io
import librosa

import pandas as pd
from transformers import BertTokenizer
import open_clip
from transformers import Wav2Vec2Processor, WhisperProcessor
from transformers import AutoImageProcessor
from transformers import VideoMAEImageProcessor, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

import h5py
import re
from PIL import Image
from io import BytesIO
import time
from scipy.stats import truncnorm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

root_data_dir = "/engram/nklab/datasets/"
cache_dir = "/engram/nklab/models/hf_cache"

tr = 1.49  # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds


modality = "all"  # @param ["visual", "audio", "language", "all"]

excluded_samples_start = 5  # @param {type:"slider", min:0, max:20, step:1}

excluded_samples_end = 5  # @param {type:"slider", min:0, max:20, step:1}

hrf_delay = 3  # @param {type:"slider", min:0, max:10, step:1}

stimulus_window = 5  # @param {type:"slider", min:1, max:20, step:1}

movies_all = [
    "friends-s01",
    "friends-s02",
    "friends-s03",
    "friends-s04",
    "friends-s05",
    "friends-s06",
    "movie10-bourne",
    "movie10-figures",
    "movie10-life",
    "movie10-wolf",
]

movies_only = ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]

last_movie = ["movie10-wolf"]

movies_train = [
    "friends-s01",
    "friends-s02",
    "friends-s03",
    "friends-s04",
    "friends-s05",
    # "movie10-bourne",
    # "movie10-figures",
    # "movie10-life",
    # "movie10-wolf",
]  # @param {allow-input: true}

ood = [
    "chaplin1",
    "chaplin2",
    "mononoke1",
    "mononoke2",
    "passepartout1",
    "passepartout2",
    "planetearth1",
    "planetearth2",
    "pulpfiction1",
    "pulpfiction2",
    "wot1",
    "wot2",
]

movies_val = ["friends-s06"]  # @param {allow-input: true}

movies_test = ["friends-s07"]  # @param {allow-input: true}


def load_fmri(root_data_dir, subject, readout_res, include_split=None, exclude_split=None):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    assert include_split is None or exclude_split is None, (
        "Cannot specify both include_split and exclude_split."
    )

    fmri = {}

    if readout_res == "parcels":
    ### Load the fMRI responses for Friends ###
    # Data directory
        fmri_file = f"sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        fmri_dir = os.path.join(
            root_data_dir,
            "algonauts_2025.competitors",
            "fmri",
            f"sub-0{subject}",
            "func",
            fmri_file,
        )
        # Load the the fMRI responses
        fmri_friends = h5py.File(fmri_dir, "r")
        for key, val in fmri_friends.items():
            fmri[str(key[13:])] = {"parcel": val[:].astype(np.float32)}
        del fmri_friends

        ### Load the fMRI responses for Movie10 ###
        # Data directory
        fmri_file = f"sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
        fmri_dir = os.path.join(
            root_data_dir,
            "algonauts_2025.competitors",
            "fmri",
            f"sub-0{subject}",
            "func",
            fmri_file,
        )
        # Load the the fMRI responses
        fmri_movie10 = h5py.File(fmri_dir, "r")
        for key, val in fmri_movie10.items():
            fmri[key[13:]] = {"parcel": val[:].astype(np.float32)}
        del fmri_movie10

        # Average the fMRI responses across the two repeats for 'figures'
        keys_all = fmri.keys()
        figures_splits = 12
        for s in range(figures_splits):
            movie = "figures" + format(s + 1, "02")
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = {
                "parcel": (
                    (fmri[keys_movie[0]]["parcel"] + fmri[keys_movie[1]]["parcel"]) / 2
                ).astype(np.float32)
            }
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]
        # Average the fMRI responses across the two repeats for 'life'
        keys_all = fmri.keys()
        life_splits = 5
        for s in range(life_splits):
            movie = "life" + format(s + 1, "02")
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = {
                "parcel": (
                    (fmri[keys_movie[0]]["parcel"] + fmri[keys_movie[1]]["parcel"]) / 2
                ).astype(np.float32)
            }
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]

        # for k, v in fmri.items():
        #     print(f"{k}: {v['parcel'].shape}")

    elif readout_res == "voxels":

        gm = "gm"
        for movie_type in ["friends", "movie10"]:
            voxel_timeseries_file = Path(
                f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/{movie_type}/Schaefer18_1000Parcels7Networks/sub-0{subject}/func/sub-0{subject}_voxel_timeseries_{gm}.h5"
            )
            voxel_timeseries = h5py.File(voxel_timeseries_file, "r")
            for name in voxel_timeseries["voxel"].keys():
                if name not in fmri:   # initialize dictionary for this name
                    fmri[name] = {}
                fmri[name]["voxel"] = voxel_timeseries["voxel"][name]

    ### Output ###
    return fmri


def extract_text(text, text_range=None):
    df = pd.DataFrame(list(text.items()), columns=["tr", "text_per_tr"])
    ### Load the transcript ###
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    ### Initialize the tokens and features lists ###
    tokens, np_tokens = [], []
    text_all = []
    # text_keys = text.keys()
    # text_values = list(text.values())

    for i in range(df.shape[0]):  # , desc="Extracting language features"):
        if text_range is not None:
            if (i<5) or (i>13):
                continue
        ### Tokenize raw text ###
        if not df.iloc[
            i
        ][
            "is_na"
        ]:  # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
            # Tokenize raw text with puntuation (for pooler_output features)
            tr_text = df.iloc[i]["text_per_tr"]
            text_all.append(tr_text)

    #         tokens.extend(tokenizer.tokenize(tr_text))
    #         # Tokenize without punctuation (for last_hidden_state features)
    #         tr_np_tokens = tokenizer.tokenize(
    #             tr_text.translate(str.maketrans('', '', string.punctuation)))
    #         np_tokens.extend(tr_np_tokens)

    # print(f"tokens: {len(tokens)}  np_tokens: {len(np_tokens)}")
    return text_all


def extract_language_context(df_transcript, ind):
    ### Initialize the tokens and features lists ###
    start = max(ind - 20, 0)
    end = ind
    df = df_transcript.iloc[start:end]["text_per_tr"]
    context_dict = df.to_dict()

    return context_dict
    # ### Loop over text chunks ###
    # for i in tqdm(range(df.shape[0]), desc="Extracting language features"):

    #     ### Tokenize raw text ###
    #     #if not df.iloc[i]["is_na"]: # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
    #     # Tokenize raw text with puntuation (for pooler_output features)
    #     tr_text = df.iloc[i]["text_per_tr"]
    #     context.extend(tr_text)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform
    
class algonauts_dataset(Dataset):
    def __init__(
        self,
        args,
        transform=None,
        include_split=None,
        exclude_split=None,
        sample_hrf=False,
    ):
        super(algonauts_dataset, self).__init__()
        # self.image_size = args.image_size
        # self.data_paths = data_paths
        self.transform = transform
        self.sample_hrf = sample_hrf
        self.num_frames = args.num_frames
        # self.is_train = is_train
        self.sub = args.subj
        # self.backbone_arch = args.backbone_arch
        self.readout_res = args.readout_res

        self.timepoints = []

        self.fmri = {}

        #     del fmris
        if include_split is not None and "s07e" in include_split:
            self.fmri = None
            target_sample_num = np.load(
                f"/engram/nklab/datasets/algonauts_2025.competitors/fmri/sub-01/target_sample_number/sub-01_friends-s7_fmri_samples.npy",
                allow_pickle=True,
            ).item()
            self.timepoints += [
                [include_split, i] for i in range(0, target_sample_num[include_split])
            ]

        elif include_split is not None and "ood" in include_split:    
            self.fmri = None
            target_sample_num = np.load(
                f"/engram/nklab/datasets/algonauts_2025.competitors/fmri/sub-01/target_sample_number/sub-01_ood_fmri_samples.npy",
                allow_pickle=True,
            ).item()

            print(f'include split: {include_split}')

            self.timepoints += [
                [include_split, i] for i in range(0, target_sample_num[include_split.split('_')[1]])
            ]
        else:
            fmris = []
            for subj in [1, 2, 3, 5]:
                fmri = load_fmri(root_data_dir, subj, self.readout_res)
                self.fmri[subj] = fmri
                fmris.append(fmri)

            for split, data in self.fmri[1].items():
                if split == 's04e01a' or split == 's04e01b' or split == 's04e13b' or split == 's05e20a' or split == 's06e03a':
                    continue

                if include_split is not None and include_split not in split:
                    continue
                elif exclude_split is not None and exclude_split in split:
                    continue

                if include_split is None  and args.objective is not None:
                    if 'finetune' in args.objective:
                        if 'movie' in args.objective:
                            if 'figures' not in split and 'life' not in split and 'wolf' not in split and 'bourne' not in split:
                                continue
                        elif 'friends' in args.objective:
                            if 's01e' not in split and 's02e' not in split and 's03e' not in split and 's04e' not in split and 's05e' not in split and 's06e' not in split:
                                continue
                        else:
                            train_s = args.objective.split('_')[1]
                            if train_s not in split:
                                continue
                # if "s06e03a" == split:

                if self.readout_res == "parcels":
                    split_len = max([len(fmri[split]["parcel"]) for fmri in fmris if split in fmri])
                elif self.readout_res == "voxels":
                    split_len = max([len(fmri[split]["voxel"]) for fmri in fmris if split in fmri])
                # else:
                #     split_len = data["parcel"].shape[0]
                self.timepoints += [[split, i] for i in range(split_len)]

            del fmris

        self.modality = args.modality

        # Load the model and tokenizer
        # language_model, self.tokenizer = get_language_model()
        # self.language_model = language_model.cuda()

        # if args.distributed == 1:
        #     self.language_model = torch.nn.parallel.DistributedDataParallel(
        #         self.language_model, device_ids=[args.gpu], find_unused_parameters=True)
        if "text" in self.modality:
            # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
            # model = GPT2Model.from_pretrained("gpt2")
            #if args.text_bb == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True, padding_side="left", cache_dir=cache_dir
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
                self.tokenizer.add_special_tokens(
                    {"pad_token": self.tokenizer.pad_token}
                )
            elif args.text_bb == 'deberta_v2_xlarge':
                model_name = "microsoft/deberta-v2-xlarge"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

            # model_name = "microsoft/deberta-v2-xxlarge"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 1. Load model + tokenizer
            model_name = 'ViT-B-14' # or 'ViT-L-14', 'ViT-H-14', etc.
            self.clip_tokenizer = open_clip.get_tokenizer(model_name, cache_dir=cache_dir)

            model_name = "meta-llama/Llama-3.2-1B"
            self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token or "[PAD]"
                self.llama_tokenizer.add_special_tokens(
                    {"pad_token": self.llama_tokenizer.pad_token}
                )

        if "audio" in self.modality:
            self.audio_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h", cache_dir=cache_dir
            )

            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir=cache_dir)

        if 'visual' in self.modality:
            self.video_feature_extractor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400", cache_dir=cache_dir)


            config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True, cache_dir=cache_dir)
            config.output_hidden_states = True
            self.video_feature_extractor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base", cache_dir=cache_dir)
            self.video_feature_extractor.do_rescale = False

            self.video_bb = args.video_bb

            if self.video_bb == 'InternVideo':
                # model setting
                self.video_transform = build_transform(224)


        self.parcellation = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.sub:02}/func/schaefer_parcellation.npy"
        )
        self.epi_mask = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.sub:02}/func/epi_mask.npy"
        )
        self.masked_parcellation = self.parcellation[self.epi_mask.astype(bool)]
        masks = np.zeros(
            (
                np.unique(self.masked_parcellation).shape[0],
                self.masked_parcellation.shape[0],
            )
        )
        for i in range(np.unique(self.masked_parcellation).shape[0]):
            masks[i, :] = (self.masked_parcellation == i).astype(bool)

        self.stim_paths = {
            "friends": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/friends_smaller.h5",
            "movie10": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/movie10_smaller.h5",
            "ood": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/ood_smaller.h5",
        }
        self.stim_files = {} 

        # self.stim = {}
        # for movie_type in ["friends", "movie10"]:
        #     f = h5py.File(
        #         f"/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/{movie_type}_compressed.h5",
        #         "r",
        #     )
        #     for key in f.keys():
        #         self.stim[key] = f[key]

                

    def subsample_frames_lazy(
        self,
        stim,
        frames,
        start,
        end,
        split,
            sample_inds=np.array(
            [
                -350,
                -320,
                -280,
                -240,
                -198,
                -185,
                -177,
                -168,
                -163,
                -158,
                -149,
                -142,
                -138,
                -134,
                -120,
                -104,
                -78,
                -61,
                -30,
                -19,
            ]
        ),
    ):
        

        if self.num_frames == 48:
            sample_inds=np.array(
            [
                -500,
                -450,
                -400,
                -390,
                -380,
                -370,
                -360,
                -350,
                -340,
                -330,
                -320,
                -310,
                -300,
                -290,
                -280,
                -270,
                -260,
                -250,
                -240,
                -230,
                -220,
                -210,
                -200,
                -195,
                -190,
                -185,
                -180,
                -175,
                -170,
                -165,
                -160,
                -155,
                -150,
                -145,
                -140,
                -135,
                -130,
                -125,
                -120,
                -110,
                -100,
                -90,
                -80,
                -70,
                -60,
                -50,
                -40,
                -30,
            ]
        ),

        if self.num_frames == 16:
            sample_inds = np.array(
                [
                    -280,
                    -240,
                    -198,
                    -185,
                    -177,
                    -168,
                    -163,
                    -158,
                    -149,
                    -142,
                    -138,
                    -134,
                    -120,
                    -104,
                    -78,
                    -61,
                ]
            )


        start, end = int(start), int(end)
        # print(f"start: {start}, end: {end}")
        slice_indices = np.arange(
            start, end
        )  # these are the valid dataset indices for the slice

        # print(f"slice_indices: {slice_indices}")

        sample_inds = np.array(sample_inds)
        corrected_sample_inds = np.where(
            sample_inds < 0, sample_inds + len(slice_indices), sample_inds
        )

        # print(f"frames.shape: {frames.shape}")
        # print(f"corrected_sample_inds: {corrected_sample_inds}")

        mask = (corrected_sample_inds >= 0) & (
            corrected_sample_inds < len(slice_indices)
        )
        corrected_sample_inds = corrected_sample_inds[mask]

        # print(f"corrected_sample_inds after mask: {corrected_sample_inds}")

        absolute_indices = slice_indices[corrected_sample_inds]

        # print(f"absolute_indices: {absolute_indices}")

        # Load each frame individually from the h5 dataset into a NumPy array
        # loaded_frames = np.array([frames[i] for i in absolute_indices])
        # t0 = time.perf_counter()
        mask = (absolute_indices >= 0) & (absolute_indices < len(frames))
        absolute_indices = absolute_indices[mask]
        loaded_frames = frames[np.unique(absolute_indices)]
        frames_processed = []
        for frame in loaded_frames:

            # with BytesIO(frame.tobytes()) as buffer:
            #     img = Image.open(buffer).convert("RGB")
            #     width, height = img.size

            #     frames_processed.append(img)
            # frame_bytes = bytes(memoryview(frame))
            img_tensor = io.decode_image(
                #torch.frombuffer(frame, dtype=torch.uint8).clone(),
                torch.tensor(frame),
                mode=io.ImageReadMode.RGB,
            )
            # [3, 224, 336]
            frames_processed.append(img_tensor)

        if len(frames_processed) < len(sample_inds):
            missing_num_frames = len(sample_inds) - len(frames_processed)
            sample_frame = io.decode_image(
                torch.tensor(stim[split]["video"][0]),
                mode=io.ImageReadMode.RGB,
            )
            padding_frame = torch.zeros(sample_frame.shape, dtype=sample_frame.dtype)
            # black_frame = Image.new("RGB", (width, height), (0, 0, 0))
            # black_frame = black_frame.clone().contiguous()
            frames_processed = [padding_frame] * missing_num_frames + frames_processed
            # frames_processed = torch.cat((padding_frames, frames_processed), dim=0)
        # print(f"h5 loading in sample_frames_lazy: {time.perf_counter() - t0} seconds")

        frames_v = {}
        if self.video_bb == 'timesformer' or self.video_bb == 'VideoMAE':
            frames = torch.stack(frames_processed).clone().contiguous()
            frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
            frames_pil = [to_pil_image(frame) for frame in frames]
            if self.num_frames == 20:
                frames_pil = frames_pil[2:18]  # Select frames from 2 to 18 (16 frames total)
            elif self.num_frames == 48:
                frames_pil = frames_pil[22:38]  # Select frames from 22 to 37 (16 frames total)

            #print(f"frames_pil: {len(frames_pil)} frames")

            # This will normalize, resize, and stack the frames into (B, C, T, H, W)
            frames_v = self.video_feature_extractor(frames_pil, return_tensors="pt")
            # print(f"frames_v shape: {frames_v['pixel_values'].shape}")

        if self.video_bb == 'InternVideo':
            frames = torch.stack(frames_processed).clone().contiguous()
            frames_pil = [to_pil_image(frame) for frame in frames]
            pixel_values = [self.video_transform(frame) for frame in frames_pil]
            pixel_values = torch.stack(pixel_values) #torch.Size([20, 3, 224, 224])
            frames_v['pixel_values'] = pixel_values
            
        return frames_processed, frames_v

    def transform_audio(self, audio, sr):
        pad_size = 492818 - audio.shape[0]

        if pad_size > 0:
            audio = np.pad(audio, (pad_size, 0), mode="constant", constant_values=0)

        # audio = audio[22050*6:22050*12]

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

        # audio_input = self.audio_processor(
        #     audio, sampling_rate=sr, return_tensors="pt", padding=True
        # )

        # audio_input = audio_input["input_values"].squeeze(0)
        return audio

    def transform_text(self, context, text=None):
        
        bert_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        text_all = extract_text(context, text_range=(5, 13))
        text = ' '.join(text_all)
        text = text if text.strip() else "[UNK]" 
        clip_tokens = self.clip_tokenizer(text)  # (batch_size, seq_len)
        #input['text_clip'] = clip_tokens

        text_all = extract_text(context)  #, text_range=(5, 13))
        text = ' '.join(text_all)
        llama_tokens = self.llama_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        return bert_tokens, clip_tokens, llama_tokens
        # print(text)
        # print(text['input_ids'].shape)


    def __getitem__(self, idx):
        split, ind = self.timepoints[idx]
        data_point = {"split": split, "ind": ind}

        # --- FMRI (unchanged) ---
        fmri_data = {}
        if self.fmri is not None:
            if self.readout_res == "parcels":
                for subj in self.fmri:
                    if len(self.fmri[subj][split]["parcel"]) > ind:
                        fmri_data[f"sub_{subj}"] = self.fmri[subj][split]["parcel"][ind]
                    else:
                        fmri_data[f"sub_{subj}"] = None
                for subj in self.fmri:
                    if fmri_data[f"sub_{subj}"] is None:
                        fmri_data[f"sub_{subj}"] = np.mean(
                            [v for v in fmri_data.values() if v is not None], axis=0
                        )

                # fmri_data = np.concatenate((
                #     fmri_data['sub_1'],
                #     fmri_data['sub_2'],
                #     fmri_data['sub_3'],
                #     fmri_data['sub_5']
                # ), axis=0)
            elif self.readout_res == "voxels":
                for subj in self.fmri:
                    if len(self.fmri[subj][split]["voxel"]) > ind:
                        fmri_data[f"sub_{subj}"] = self.fmri[subj][split]["voxel"][ind]
                    else:
                        fmri_data[f"sub_{subj}"] = None
                for subj in self.fmri:
                    if fmri_data[f"sub_{subj}"] is None:
                        fmri_data[f"sub_{subj}"] = np.mean(
                            [v for v in fmri_data.values() if v is not None], axis=0
                        )

                    fmri_data[f"sub_{subj}"] = fmri_data[f"sub_{subj}"].astype(np.float32)
                # fmri_data = np.concatenate((
                #     fmri_data['sub_1'],
                #     fmri_data['sub_2'],
                #     fmri_data['sub_3'],
                #     fmri_data['sub_5']
                # ), axis=0)

        # --- Text (unchanged) ---
        # TODO part of this should not repeat for every sample
        if 'chaplin' not in split:

            if split.split("_")[0] == "ood":
                transcript_path = (f'{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/ood/{split.split("_")[1][:-1]}/{split}.tsv')
            else:
                transcript_path = (
                    f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/"
                    + (f"friends/s{split[2]}/friends_{split}.tsv"
                    if re.match(r"^s\d\de\d\d[a-zA-Z]$", split)
                    else f"movie10/{split[:-2]}/movie10_{split}.tsv")
                )


            transcript_df = pd.read_csv(transcript_path, sep="\t")
            context = extract_language_context(transcript_df, ind)
            

            text_all = extract_text(context)
            text = " ".join(text_all)
            text = text if text.strip() else "[UNK]"


        if "text" in self.modality:
            if 'chaplin' in split:
                text = '   '

            bert_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            llama_tokens = self.llama_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            print(f"Context for split {split}, ind {ind}:")
            print(f"Original text: {text}")
            if 'chaplin' not in split:
                text_all = extract_text(context, text_range=(5, 13))
                text = ' '.join(text_all)
                text = text if text.strip() else "[UNK]" 

                print(f"CLIP text: {text}")

            clip_tokens = self.clip_tokenizer(text)  # (batch_size, seq_len)
            #input['text_clip'] = clip_tokens

                #bert_tokens, clip_tokens, llama_tokens = self.transform_text(context)
            data_point["text"] = bert_tokens
            data_point["text_clip"] = clip_tokens
            data_point["text_llama"] = llama_tokens

        # --- Open h5py file INSIDE __getitem__ ---
        if split.split("_")[0] == "ood":
            movie_type = "ood"
        else:
            movie_type = "friends" if "s0" in split[:2] else "movie10"
        stim_path = self.stim_paths[movie_type]

        with h5py.File(stim_path, "r") as stim:
            # --- Video ---
            if split.split("_")[0] == "ood":
                split = split.split("_")[1]

            video_fps = stim[split]["video"].attrs["fps"]
            end = int((ind + 1) * tr * video_fps)
            start = max(int((ind - 14) * tr * video_fps), 0)
            frames_ds = stim[split]["video"]

            if self.sample_hrf:
                mean, std = -150, 50
                lower, upper = -669, 0
                all_frames = np.arange(lower, upper)
                a, b = (lower - mean) / std, (upper - mean) / std
                probs = truncnorm.pdf(all_frames, a, b, loc=mean, scale=std)
                probs /= probs.sum()
                sample_inds = np.random.choice(all_frames, size=self.num_frames, p=probs, replace=False)
                frames, frames_v = self.subsample_frames_lazy(stim, frames_ds, start, end, split, sample_inds)
            else:
                frames, frames_v = self.subsample_frames_lazy(stim, frames_ds, start, end, split)

            if "visual" in self.modality and self.transform:
                frames = torch.stack([self.transform(frame) for frame in frames])
                data_point["visual"] = frames

                if self.video_bb != 'None':
                    print('self.vbb',self.video_bb)
                    data_point["video"] = frames_v['pixel_values']  # (B, C, T, H, W)
                #.permute(0, 2, 1, 3, 4)
            # --- Audio ---
            sr = stim[split]["audio"].attrs["sr"]
            end_audio = int((ind + 1) * tr * sr)
            start_audio = max(int((ind - 14) * tr * sr), 0)
            audio = stim[split]["audio"][start_audio:end_audio]

        if "audio" in self.modality:
            data_point["audio"] = self.transform_audio(audio, sr)
            data_point["sr"] = 16000  # if you're resampling

        return data_point, fmri_data


    def __len__(self):
        return len(self.timepoints)





