import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torchvision.io as io
import librosa

import pandas as pd
from transformers import Wav2Vec2Processor, WhisperProcessor
from transformers import AutoImageProcessor
from transformers import VideoMAEImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append("/engram/nklab/pf2477")
from multimodal_encoder.models.multimodel_backbone import ProcessorWrapper, FeatureWrapper

import h5py
import re
from scipy.stats import truncnorm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

root_data_dir = "/engram/nklab/datasets/"
cache_dir = "/engram/nklab/models/hf_cache"

tr = 1.49  # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds

modality = "all"  # @param ["video", "audio", "language", "all"]

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

FRIENDS_SEASONS = ["s01", "s02", "s03", "s04", "s05", "s06"]
MOVIE10_NAMES = ["bourne", "figures", "life", "wolf"]

SPLIT_GROUP_ALIASES = {
    "friends": FRIENDS_SEASONS,
    "all-friends": FRIENDS_SEASONS,
    "movie10": MOVIE10_NAMES,
    "all-movie10": MOVIE10_NAMES,
    "friends-train-default": ["s01", "s02", "s03", "s04", "s05"],
    "friends-test-default": ["s06"],
    "movie10-ood-default": MOVIE10_NAMES,
    "movie10-train-default": ["bourne", "figures", "life"],
    "movie10-test-default": ["wolf"],
    "friends-ood-default": FRIENDS_SEASONS,
}


def _normalize_split_spec(spec):
    """Convert split specification into a normalized token list.

    Accepts `None`, comma-separated strings, or list/tuple/set of strings.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        return [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if isinstance(spec, (list, tuple, set)):
        return [str(tok).strip().lower() for tok in spec if str(tok).strip()]
    raise TypeError(f"Unsupported split specification type: {type(spec)}")


def _split_matches_token(split, token):
    """Flexible matching between a concrete split name and a user token."""
    split = split.lower()
    token = token.lower().strip()

    # recursively expand aliases
    if token in SPLIT_GROUP_ALIASES:
        return any(_split_matches_token(split, t) for t in SPLIT_GROUP_ALIASES[token])

    # normalized dataset-level names
    if token.startswith("friends-"):
        token = token.split("-", 1)[1]
    if token.startswith("movie10-"):
        token = token.split("-", 1)[1]

    # season-level friend selector (e.g., s06)
    if re.fullmatch(r"s\d{2}", token):
        return split.startswith(f"{token}e")

    # movie-level selector (e.g., life)
    if token in MOVIE10_NAMES:
        return token in split

    # fallback: substring match for custom patterns (e.g., s06e03a)
    return token in split


def _split_is_selected(split, include_tokens, exclude_tokens):
    if include_tokens and not any(_split_matches_token(split, tok) for tok in include_tokens):
        return False
    if exclude_tokens and any(_split_matches_token(split, tok) for tok in exclude_tokens):
        return False
    return True


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
        with h5py.File(fmri_dir, "r") as fmri_friends:
            for key, val in fmri_friends.items():
                fmri[str(key[13:])] = {"parcel": np.asarray(val, dtype=np.float32)}

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
        with h5py.File(fmri_dir, "r") as fmri_movie10:
            for key, val in fmri_movie10.items():
                fmri[key[13:]] = {"parcel": np.asarray(val, dtype=np.float32)}

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

    elif readout_res == "voxels":
        gm = "gm"
        for movie_type in ["friends", "movie10"]:
            voxel_timeseries_file = Path(
                f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/{movie_type}/Schaefer18_1000Parcels7Networks/sub-0{subject}/func/sub-0{subject}_voxel_timeseries_{gm}.h5"
            )
            with h5py.File(voxel_timeseries_file, "r") as voxel_timeseries:
                # print(list(voxel_timeseries["voxel"].keys())) # episode splits
                for name in voxel_timeseries["voxel"].keys():
                    if name not in fmri:   # initialize dictionary for this name
                        fmri[name] = {}
                    fmri[name]["voxel"] = np.asarray(
                        voxel_timeseries["voxel"][name], dtype=np.float32
                    )

    ### Output ###
    return fmri


def extract_text(text, text_range=None):
    df = pd.DataFrame(list(text.items()), columns=["tr", "text_per_tr"])
    ### Load the transcript ###
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    ### Initialize the features list ###
    text_all = []

    for i in range(df.shape[0]):  # , desc="Extracting language features"):
        if text_range is not None:
            if (i<5) or (i>13):   # for clip model to narrow down the window
                continue
        ### Tokenize raw text ###
        if not df.iloc[i]["is_na"]:  # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
            # Tokenize raw text with puntuation (for pooler_output features)
            tr_text = df.iloc[i]["text_per_tr"]
            text_all.append(tr_text)

    # print(f"tokens: {len(tokens)}  np_tokens: {len(np_tokens)}")
    return text_all


def extract_language_context(df_movie, ind):
    ### Initialize the tokens and features lists ###
    start = max(ind - 20, 0)
    end = ind
    df = df_movie.iloc[start:end]["text_per_tr"]
    context_dict = df.to_dict()

    return context_dict

def build_transform(input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=mean, std=std)]
    )
    return transform


class algonauts_dataset(Dataset):
    def __init__(
        self,
        args,
        transform=None,
        include_split=None,
        exclude_split=None,
        include_splits=None,
        exclude_splits=None,
        sample_hrf=False
    ):
        super(algonauts_dataset, self).__init__()
        self.transform = transform
        self.sample_hrf = sample_hrf
        self.num_frames = args.num_frames
        self.readout_res = args.readout_res

        self.timepoints = []
        self.samples: List[Dict[str, object]] = []

        self.subj = args.subj
        self.fmri = {}

        self.backbone_list = args.backbone_list

        # Subject selection:
        #   args.subj == 0 -> load all available subjects (default behavior)
        #   args.subj > 0  -> load only that subject
        all_subject_ids = [1, 2, 3, 5]
        if self.subj == 0:
            self.subject_ids = all_subject_ids
        else:
            if self.subj not in all_subject_ids:
                raise ValueError(
                    f"Unsupported subject id: {self.subj}. Use 0 for all subjects or one of {all_subject_ids}."
                )
            self.subject_ids = [self.subj]

        # Backward-compatible single-string arguments and new list-style arguments.
        include_tokens = _normalize_split_spec(include_splits if include_splits is not None else include_split)
        exclude_tokens = _normalize_split_spec(exclude_splits if exclude_splits is not None else exclude_split)

        fmris = []
        for subj in self.subject_ids:
            fmri = load_fmri(root_data_dir, subj, self.readout_res)
            self.fmri[subj] = fmri
            fmris.append(fmri)

        # Known problematic chunks, kept excluded by default.
        self.always_excluded_splits = {"s04e01a", "s04e01b", "s04e13b", "s05e20a", "s06e03a"}
        # Used for subject-specific resources that require a single id (e.g., parcellation/mask files).
        self.reference_subj = self.subject_ids[0]
        for split in self.fmri[self.reference_subj].keys():
            if split in self.always_excluded_splits or not _split_is_selected(split, include_tokens, exclude_tokens):
                continue

            if self.readout_res == "parcels":
                split_len = max([len(fmri[split]["parcel"]) for fmri in fmris if split in fmri])
            elif self.readout_res == "voxels":
                split_len = max([len(fmri[split]["voxel"]) for fmri in fmris if split in fmri])
            for i in range(split_len):

                if self.readout_res == "parcels":
                    available_subjects = [
                        subj for subj in self.subject_ids
                        if split in self.fmri[subj] and len(self.fmri[subj][split]["parcel"]) > i
                    ]
                else:
                    available_subjects = [
                        subj for subj in self.subject_ids
                        if split in self.fmri[subj] and len(self.fmri[subj][split]["voxel"]) > i
                    ]

                if len(available_subjects) == 0:
                    continue
                self.timepoints.append([split, i])
                self.samples.append(
                    {
                        "split": split,
                        "ind": i,
                        "available_subjects": available_subjects,
                        "has_missing_targets": len(available_subjects) != len(self.subject_ids),
                    }
                )

        del fmris

        self.modality = args.modality

        if "text" in self.modality:
            text_backbone = getattr(args, "text_backbone", "metaclip")
            text_backbone = str(text_backbone).lower()
            self.text_backbone = text_backbone

            if text_backbone not in self.backbone_list['text']:
                raise ValueError(
                    f"Unsupported text backbone: {text_backbone}. "
                    f"Supported options are: {sorted(self.backbone_list['text'])}."
                )

            self.text_processor = ProcessorWrapper(
                modality="text",
                backbone=text_backbone,
                cache_dir=cache_dir,
            )

        if "audio" in self.modality:
            # Support either `args.audio_backbone` (new) or `args.audio_bb` (legacy).
            audio_backbone = getattr(args, "audio_backbone", "whisper")
            audio_backbone = str(audio_backbone).lower()
            self.audio_backbone = audio_backbone

            if audio_backbone not in self.backbone_list['audio']:
                raise ValueError(
                    f"Unsupported audio backbone: {audio_backbone}. "
                    f"Supported options are: {sorted(self.backbone_list['audio'])}."
                )

            self.audio_processor = ProcessorWrapper(
                modality="audio",
                backbone=audio_backbone,
                cache_dir=cache_dir,
            )

        if "video" in self.modality:
            # Support either `args.video_backbone` (new) or `args.video_bb` (legacy).
            video_backbone = getattr(args, "video_backbone", "dino")
            video_backbone = str(video_backbone).lower()
            self.video_backbone = video_backbone

            if video_backbone not in self.backbone_list['video']:
                raise ValueError(
                    f"Unsupported video backbone: {video_backbone}. "
                    f"Supported options are: {sorted(self.backbone_list['video'])}."
                )
            
            self.video_processor = ProcessorWrapper(
                modality="video",
                backbone=video_backbone,
                cache_dir=cache_dir,
            )

            # Temporal length expected by video transformers (configurable).
            self.video_target_frames = int(getattr(args, "video_target_frames", 16))

        self.parcellation = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.reference_subj:02}/func/schaefer_parcellation.npy"
        )
        self.epi_mask = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.reference_subj:02}/func/epi_mask.npy"
        )
        self.masked_parcellation = self.parcellation[self.epi_mask.astype(bool)]

        self.stim_paths = {
            "friends": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/friends_smaller.h5",
            "movie10": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/movie10_smaller.h5",
        }

        self.transcript_paths: Dict[str, str] = {}
        self._transcript_cache: Dict[str, pd.DataFrame] = {}
        for sample in self.samples:
            split = str(sample["split"])
            if split in self.transcript_paths:
                continue
            self.transcript_paths[split] = (
                f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/"
                + (f"friends/s{split[2]}/friends_{split}.tsv"
                if re.match(r"^s\d\de\d\d[a-zA-Z]$", split)
                else f"movie10/{split[:-2]}/movie10_{split}.tsv")
            )

        # Per-process worker-local HDF5 handles, lazily opened to avoid repeated open/close overhead.
        self._stim_handles: Dict[Tuple[int, str], h5py.File] = {}

        # Explicit compatibility behavior for missing per-subject targets.
        # "mean_fill" preserves legacy behavior; "strict" raises an error for traceability.
        self.missing_target_policy = str(getattr(args, "missing_target_policy", "mean_fill")).lower()
        if self.missing_target_policy not in {"mean_fill", "strict"}:
            raise ValueError(
                f"Unsupported missing_target_policy: {self.missing_target_policy}. "
                "Use one of ['mean_fill', 'strict']."
            )

    def _get_transcript_df(self, split: str) -> pd.DataFrame:
        transcript_df = self._transcript_cache.get(split)
        if transcript_df is None:
            transcript_df = pd.read_csv(self.transcript_paths[split], sep="\t")
            self._transcript_cache[split] = transcript_df
        return transcript_df

    def _movie_type_from_split(self, split: str) -> str:
        return "friends" if re.match(r"^s\d\de\d\d[a-zA-Z]$", split) else "movie10"

    def _get_stim_handle(self, split: str):
        movie_type = self._movie_type_from_split(split)
        pid = os.getpid()
        key = (pid, movie_type)
        handle = self._stim_handles.get(key)
        if handle is None:
            handle = h5py.File(self.stim_paths[movie_type], "r")
            self._stim_handles[key] = handle
        return handle

    def _collect_fmri_targets(self, split: str, ind: int, available_subjects: List[int]):
        fmri_data = {}
        value_key = "parcel" if self.readout_res == "parcels" else "voxel"

        available_arrays = []
        for subj in available_subjects:
            value = self.fmri[subj][split][value_key][ind]
            if value is not None:
                available_arrays.append(np.asarray(value, dtype=np.float32))

        if len(available_arrays) == 0:
            raise RuntimeError(f"No available targets for split={split}, ind={ind}")

        fill_value = np.mean(available_arrays, axis=0).astype(np.float32)

        for subj in self.subject_ids:
            if subj in available_subjects:
                value = self.fmri[subj][split][value_key][ind]
                fmri_data[f"sub_{subj}"] = np.asarray(value, dtype=np.float32)
            else:
                if self.missing_target_policy == "strict":
                    raise RuntimeError(
                        f"Missing target for subject={subj}, split={split}, ind={ind} "
                        f"under missing_target_policy='strict'."
                    )
                fmri_data[f"sub_{subj}"] = fill_value

        return fmri_data

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stim_handles"] = {}
        return state

    def __del__(self):
        for handle in self._stim_handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def subsample_frames_lazy(
        self, stim, frames, start, end, split,
        sample_inds=np.array([
            -350, -320, -280, -240, -198, -185, -177, -168, -163, -158, -149, -142, -138, -134, -120, -104, -78, -61, -30, -19,
        ]),
    ):
        if self.num_frames == 48:
            sample_inds = np.array([
                -500, -450, -400, -390, -380, -370, -360, -350, -340, -330, -320, -310, -300, -290, -280, -270,
                -260, -250, -240, -230, -220, -210, -200, -195, -190, -185, -180, -175, -170, -165, -160, -155,
                -150, -145, -140, -135, -130, -125, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30,
            ])

        if self.num_frames == 16:
            sample_inds = np.array([
                -280, -240, -198, -185, -177, -168, -163, -158, -149, -142, -138, -134, -120, -104, -78, -61,
            ])

        start, end = int(start), int(end)
        slice_indices = np.arange(start, end)  # these are the valid dataset indices for the slice

        sample_inds = np.array(sample_inds)
        corrected_sample_inds = np.where(
            sample_inds < 0, sample_inds + len(slice_indices), sample_inds
        )

        mask = (corrected_sample_inds >= 0) & (
            corrected_sample_inds < len(slice_indices)
        )
        corrected_sample_inds = corrected_sample_inds[mask]

        absolute_indices = slice_indices[corrected_sample_inds]

        # Load each frame individually from the h5 dataset into a NumPy array
        mask = (absolute_indices >= 0) & (absolute_indices < len(frames))
        absolute_indices = absolute_indices[mask]
        loaded_frames = frames[np.unique(absolute_indices)]
        frames_processed = []
        for frame in loaded_frames:
            img_tensor = io.decode_image(
                torch.tensor(frame),
                mode=io.ImageReadMode.RGB,
            )# [3, 224, 336]
            frames_processed.append(img_tensor)

        if len(frames_processed) < len(sample_inds):
            missing_num_frames = len(sample_inds) - len(frames_processed)
            sample_frame = io.decode_image(
                torch.tensor(stim[split]["video"][0]),
                mode=io.ImageReadMode.RGB,
            )
            padding_frame = torch.zeros(sample_frame.shape, dtype=sample_frame.dtype)
            frames_processed = [padding_frame] * missing_num_frames + frames_processed
        # print(f"h5 loading in sample_frames_lazy: {time.perf_counter() - t0} seconds")

        # `frames_processed`: decoded raw RGB frames as tensors [3, H, W] per time step.
        frames = torch.stack(frames_processed).clone().contiguous()
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        frames_pil = [to_pil_image(frame) for frame in frames]
        target_t = max(1, getattr(self, "video_target_frames", 16))
        n_frames = len(frames_pil)
        if n_frames > target_t:
            # Uniform temporal subsampling for arbitrary source lengths.
            keep_idx = np.linspace(0, n_frames - 1, target_t).round().astype(int)
            frames_pil = [frames_pil[i] for i in keep_idx]
        elif n_frames < target_t and n_frames > 0:
            # Pad by repeating the last frame to meet model expected length.
            frames_pil = frames_pil + [frames_pil[-1]] * (target_t - n_frames)

        #print(f"frames_pil: {len(frames_pil)} frames")
        inputs = self.video_processor.process(frames_pil)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
        return inputs

    def transform_audio(self, audio, sr):
        # 1) resample first. Both Wav2Vec2 and Whisper are trained assuming 16 kHz audio.
        target_sr = 16000
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        target_len = 492818

        # 2) then pad or truncate in the target sample rate domain
        if audio.shape[0] < target_len:
            pad_size = target_len - audio.shape[0]
            audio = np.pad(audio, (0, pad_size), mode="constant", constant_values=0)
        else:
            audio = audio[:target_len]

        inputs = self.audio_processor.process(audio, sampling_rate=target_sr)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __getitem__(self, idx):
        sample = self.samples[idx]
        split = str(sample["split"])
        ind = int(sample["ind"])
        available_subjects = list(sample["available_subjects"])
        data_point = {"split": split, "ind": ind}

        # --- FMRI (explicit/traceable target policy) ---
        fmri_data = self._collect_fmri_targets(split, ind, available_subjects)

        if "text" in self.modality:
            # --- Text (cached transcript table per split) ---
            transcript_df = self._get_transcript_df(split)
            context = extract_language_context(transcript_df, ind)

            text_all = extract_text(context)
            text = " ".join(text_all)
            text = text if text.strip() else "[UNK]"

            if self.text_backbone in ["metaclip"]:
                # Clip only allows for a certain number of tokens and then cuts off the text
                text_all = extract_text(context, text_range=(5, 13)) # 
                text_clip = ' '.join(text_all)
                text_clip = text_clip if text_clip.strip() else "[UNK]"

                # print(f"Split: {split}  Ind: {ind}")
                # print(f"Original text: {text}")
                # print(f"CLIP text: {text_clip}")

                text_inputs = self.text_processor.process(text_clip)
            else:
                text_inputs = self.text_processor.process(text)

            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items() }
            data_point["text"] = text_inputs

        if ("video" in self.modality) or ("audio" in self.modality):
            # --- Worker-safe lazy HDF5 handles ---
            stim = self._get_stim_handle(split)

            if "video" in self.modality:
                # --- Video ---
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
                    video_inputs = self.subsample_frames_lazy(stim, frames_ds, start, end, split, sample_inds)
                else:
                    video_inputs = self.subsample_frames_lazy(stim, frames_ds, start, end, split)

                data_point["video"] = video_inputs

            if "audio" in self.modality:
                # --- Audio ---
                sr = stim[split]["audio"].attrs["sr"]
                end_audio = int((ind + 1) * tr * sr)
                start_audio = max(int((ind - 14) * tr * sr), 0)
                audio = np.asarray(stim[split]["audio"][start_audio:end_audio])
                audio_inputs = self.transform_audio(audio, sr)
                data_point["audio"] = audio_inputs

        return data_point, fmri_data


    def __len__(self):
        return len(self.samples)





