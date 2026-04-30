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
from multimodal_encoder.models.multimodel_backbone import ProcessorWrapper, FeatureWrapper, video2frames_defaults

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

MOVIE10_OOD_NAMES = [
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
MOVIE10_OOD_SET = set(MOVIE10_OOD_NAMES)

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
    "friends-challenge-default": ["s07"],
    "movie10-challenge-default": MOVIE10_OOD_NAMES,
    "movie10-attn-probing":["figures"]
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

    # explicit friends challenge selector
    if token in {"s7", "s07"}:
        return split.startswith("s07e")

    # ood selectors
    if token == "ood":
        if split.startswith("ood_"):
            return split[4:] in MOVIE10_OOD_SET # remove prefix
        return split in MOVIE10_OOD_SET

    if token.startswith("ood_"):
        token = token[4:]

    if token in MOVIE10_OOD_SET:
        return split == token or split == f"ood_{token}"

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


def _movie_type_from_split_name(split: str) -> str:
    split = split.lower()
    if re.match(r"^s\d\de\d\d[a-zA-Z]$", split):
        return "friends"
    if split.startswith("ood_"):
        return "ood"
    if split in MOVIE10_OOD_SET:
        return "ood"
    return "movie10"


def _expand_split_tokens(tokens: List[str]) -> List[str]:
    """Expand alias tokens recursively into leaf tokens."""
    expanded: List[str] = []
    stack = list(tokens)
    while stack:
        token = stack.pop()
        if token in SPLIT_GROUP_ALIASES:
            stack.extend([str(t).lower() for t in SPLIT_GROUP_ALIASES[token]])
        else:
            expanded.append(token)
    return expanded


def _token_is_no_target_leaf(token: str) -> bool:
    token = token.lower().strip()
    if token in {"s7", "s07", "ood"}:
        return True
    if re.fullmatch(r"s07e\d\d[a-zA-Z]", token):
        return True
    if token.startswith("ood_"):
        token = token[4:]
    return token in MOVIE10_OOD_SET


def _tokens_request_only_no_target(include_tokens: List[str]) -> bool:
    if not include_tokens:
        return False
    leaves = _expand_split_tokens(include_tokens)
    return len(leaves) > 0 and all(_token_is_no_target_leaf(tok) for tok in leaves)


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


def load_voxel_index(subject, include_split=None, exclude_split=None):
    """Build lightweight voxel metadata index (no eager voxel array loading)."""
    assert include_split is None or exclude_split is None, (
        "Cannot specify both include_split and exclude_split."
    )

    include_tokens = _normalize_split_spec(include_split)
    exclude_tokens = _normalize_split_spec(exclude_split)

    fmri = {}
    gm = "gm"
    for movie_type in ["friends", "movie10"]:
        voxel_timeseries_file = Path(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/{movie_type}/Schaefer18_1000Parcels7Networks/sub-0{subject}/func/sub-0{subject}_voxel_timeseries_{gm}.h5"
        )
        with h5py.File(voxel_timeseries_file, "r") as voxel_timeseries:
            for name in voxel_timeseries["voxel"].keys():
                if not _split_is_selected(name, include_tokens, exclude_tokens):
                    continue
                fmri[name] = {
                    "voxel_len": int(voxel_timeseries["voxel"][name].shape[0]),
                    "movie_type": movie_type,
                }

    return fmri


# def extract_text(text, text_processor, text_range=None, silence_mode="boundary"):
#     df = pd.DataFrame(list(text.items()), columns=["tr", "text_per_tr"])
#     ### Load the transcript ###
#     df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

#     ### Initialize the features list ###
#     text_all = []

#     for i in range(df.shape[0]):  # , desc="Extracting language features"):
#         if text_range is not None:
#             start, end = text_range
#             if i < start or i > end:   # for clip model to narrow down the window
#                 continue
#         ### Tokenize raw text ###
#         if not df.iloc[i]["is_na"]:  # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
#             # Tokenize raw text with puntuation (for pooler_output features)
#             tr_text = df.iloc[i]["text_per_tr"]
#         # else:
#         #     tr_text = text_processor.processor.eos_token
#             text_all.append(tr_text)

#     # print(f"tokens: {len(tokens)}  np_tokens: {len(np_tokens)}")
#     return text_all

def set_silence_token(text_processor):
    '''transcript without spoken words, prioritizing special tokens defined in the tokenizer.'''
    tokenizer = getattr(text_processor, "processor", text_processor)
    silence_token = getattr(tokenizer, "sep_token", None) or getattr(tokenizer, "eos_token", None)
    return silence_token

def extract_text(
    text,
    silence_token,
    text_range=None,
    silence_mode: str = "boundary",  # "skip" | "boundary" | "empty"
):
    items = list(text.items())

    if text_range is not None:
        start, end = text_range
        items = items[start:end + 1]

    text_all = []
    for _, tr_text in items:
        is_silent = (tr_text is None) or (isinstance(tr_text, str) and tr_text.strip() == "")

        if is_silent:
            if silence_mode == "skip":
                continue

            elif silence_mode == "boundary":
                if silence_token is not None:
                    text_all.append(silence_token)
                # if no token exists → effectively skip

            elif silence_mode == "empty":
                # preserve position but no semantic token
                text_all.append("")

            else:
                raise ValueError(f"Unknown silence_mode: {silence_mode}")

        else:
            text_all.append(str(tr_text).strip())

    if not text_all and silence_mode != "skip" and silence_token is not None:
        text_all.append(silence_token)
    # print(f"Extracted text: {text_all}")
    return text_all 


def extract_language_context(df_movie, ind):
    ### Initialize the tokens and features lists ###
    start = max(ind - 20, 0)
    end = ind
    df = df_movie.iloc[start:end]["text_per_tr"]
    # context_dict = df.to_dict()
    context_dict = {k: (None if pd.isna(v) else v) for k, v in df.items()}

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
        self.voxel_h5_paths: Dict[int, Dict[str, Path]] = {}

        self.stim_paths = {
            "friends": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/friends_smaller.h5",
            "movie10": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/movie10_smaller.h5",
            "ood": "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/ood_smaller.h5",
        }

        self.backbone_list = args.backbone_list

        # Subject selection
        all_subject_ids = [1, 2, 3, 5]
        if self.subj not in all_subject_ids:
            raise ValueError(
                f"Unsupported subject id: {self.subj}. Use 0 for all subjects or one of {all_subject_ids}."
            )
        self.subject_ids = [self.subj]

        # Backward-compatible single-string arguments and new list-style arguments.
        include_tokens = _normalize_split_spec(include_splits if include_splits is not None else include_split)
        exclude_tokens = _normalize_split_spec(exclude_splits if exclude_splits is not None else exclude_split)

        fmris = []
        load_targets = not _tokens_request_only_no_target(include_tokens)
        if load_targets:
            for subj in self.subject_ids:
                if self.readout_res == "voxels":
                    fmri = load_voxel_index(
                        subj,
                        include_split=include_tokens if include_tokens else None,
                        exclude_split=exclude_tokens if exclude_tokens else None,
                    )
                    gm = "gm"
                    self.voxel_h5_paths[subj] = {
                        "friends": Path(
                            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/friends/Schaefer18_1000Parcels7Networks/sub-0{subj}/func/sub-0{subj}_voxel_timeseries_{gm}.h5"
                        ),
                        "movie10": Path(
                            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-0{subj}/func/sub-0{subj}_voxel_timeseries_{gm}.h5"
                        ),
                    }
                else:
                    fmri = load_fmri(root_data_dir, subj, self.readout_res)
                self.fmri[subj] = fmri
                fmris.append(fmri)
        else:
            for subj in self.subject_ids:
                self.fmri[subj] = {}

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
                split_len = max([fmri[split]["voxel_len"] for fmri in fmris if split in fmri])
            for i in range(split_len):

                if self.readout_res == "parcels":
                    available_subjects = [
                        subj for subj in self.subject_ids
                        if split in self.fmri[subj] and len(self.fmri[subj][split]["parcel"]) > i
                    ]
                else:
                    available_subjects = [
                        subj for subj in self.subject_ids
                        if split in self.fmri[subj] and self.fmri[subj][split]["voxel_len"] > i
                    ]

                if len(available_subjects) == 0:
                    continue
                self.timepoints.append([split, i])
                self.samples.append(
                    {
                        "split": split,
                        "ind": i,
                        "available_subjects": available_subjects,
                        "has_targets": True,
                        "has_missing_targets": len(available_subjects) != len(self.subject_ids),
                    }
                )

        self._append_no_target_samples(include_tokens, exclude_tokens)

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

            self.text_silence_token = set_silence_token(self.text_processor)

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
            self.video_target_frames = video2frames_defaults.get(video_backbone, 16)

        self.parcellation = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.reference_subj:02}/func/schaefer_parcellation.npy"
        )
        self.epi_mask = np.load(
            f"/engram/nklab/eh2976/cneuromod_extract_tseries/outputs/movie10/Schaefer18_1000Parcels7Networks/sub-{self.reference_subj:02}/func/epi_mask.npy"
        )
        self.masked_parcellation = self.parcellation[self.epi_mask.astype(bool)]
        self.valid_voxel_mask = self.masked_parcellation > 0
        # if self.readout_res == "voxels":
        #     self.masked_parcellation = self.masked_parcellation[self.valid_voxel_mask]
        #     if self.masked_parcellation.size == 0:
        #         raise ValueError("No valid voxels found after filtering masked_parcellation > 0.")
        #     self.num_valid_voxels = int(self.masked_parcellation.shape[0])
        # else:
        #     self.num_valid_voxels = int(self.masked_parcellation.shape[0])

        self.transcript_paths: Dict[str, str] = {}
        self._transcript_cache: Dict[str, pd.DataFrame] = {}
        for sample in self.samples:
            split = str(sample["split"])
            if split in self.transcript_paths:
                continue
            transcript_path = self._resolve_transcript_path(split)
            if transcript_path is not None:
                self.transcript_paths[split] = transcript_path

        # Per-process worker-local HDF5 handles, lazily opened to avoid repeated open/close overhead.
        self._stim_handles: Dict[Tuple[int, str], h5py.File] = {}
        self._voxel_handles: Dict[Tuple[int, int, str], h5py.File] = {}

        # Explicit compatibility behavior for missing per-subject targets.
        # "mean_fill" preserves legacy behavior; "strict" raises an error for traceability.
        self.missing_target_policy = str(getattr(args, "missing_target_policy", "mean_fill")).lower()
        if self.missing_target_policy not in {"mean_fill", "strict"}:
            raise ValueError(
                f"Unsupported missing_target_policy: {self.missing_target_policy}. "
                "Use one of ['mean_fill', 'strict']."
            )

    def _load_sample_count_file(self, kind: str) -> Dict[str, int]:
        if kind == "s7":
            path = (
                f"/engram/nklab/datasets/algonauts_2025.competitors/fmri/sub-0{self.subj}/target_sample_number/"
                f"sub-0{self.subj}_friends-s7_fmri_samples.npy"
            )
        elif kind == "ood":
            path = (
                f"/engram/nklab/datasets/algonauts_2025.competitors/fmri/sub-0{self.subj}/target_sample_number/"
                f"sub-0{self.subj}_ood_fmri_samples.npy"
            )
        else:
            raise ValueError(f"Unsupported sample count kind: {kind}")

        if not os.path.exists(path):
            return {}

        data = np.load(path, allow_pickle=True).item()
        return {str(k): int(v) for k, v in data.items()}

    def _resolve_requested_no_target_splits(
        self,
        include_tokens: List[str],
        exclude_tokens: List[str],
    ) -> List[str]:
        if not include_tokens:
            return []

        selected: List[str] = []

        s7_counts = self._load_sample_count_file("s7")
        for split in sorted(s7_counts.keys()):
            if split in self.always_excluded_splits:
                continue
            if _split_is_selected(split, include_tokens, exclude_tokens):
                selected.append(split)

        ood_counts = self._load_sample_count_file("ood")
        for name in sorted(ood_counts.keys()):
            split = f"ood_{name}"
            if _split_is_selected(split, include_tokens, exclude_tokens):
                selected.append(split)

        return selected

    def _infer_no_target_split_length_from_stim(self, split: str) -> int:
        movie_type = self._movie_type_from_split(split)
        stim_key = self._stim_split_key(split)
        with h5py.File(self.stim_paths[movie_type], "r") as stim:
            if stim_key not in stim:
                raise KeyError(
                    f"Split '{stim_key}' not found in {self.stim_paths[movie_type]} "
                    f"(requested as '{split}')."
                )
            group = stim[stim_key]
            if "audio" in group:
                return int(group["audio"].shape[0])
            if "video" in group:
                return int(group["video"].shape[0])
        raise RuntimeError(f"Unable to infer sample count for no-target split '{split}'.")

    def _append_no_target_samples(self, include_tokens: List[str], exclude_tokens: List[str]) -> None:
        requested = self._resolve_requested_no_target_splits(include_tokens, exclude_tokens)
        if not requested:
            return

        s7_counts = self._load_sample_count_file("s7")
        ood_counts = self._load_sample_count_file("ood")

        existing = {(str(sample["split"]), int(sample["ind"])) for sample in self.samples}

        for split in requested:
            if split.startswith("s07e"):
                split_len = int(s7_counts.get(split, 0))
            elif split.startswith("ood_"):
                split_len = int(ood_counts.get(split.split("_", 1)[1], 0))
            else:
                split_len = 0

            if split_len <= 0:
                split_len = self._infer_no_target_split_length_from_stim(split)

            for i in range(split_len):
                key = (split, i)
                if key in existing:
                    continue
                self.timepoints.append([split, i])
                self.samples.append(
                    {
                        "split": split,
                        "ind": i,
                        "available_subjects": [],
                        "has_targets": False,
                        "has_missing_targets": False,
                    }
                )
                existing.add(key)

    def _resolve_transcript_path(self, split: str) -> Optional[str]:
        transcripts_root = f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts"
        movie_type = self._movie_type_from_split(split)

        if movie_type == "friends":
            path = f"{transcripts_root}/friends/s{split[2]}/friends_{split}.tsv"
            return path if os.path.exists(path) else None

        if movie_type == "ood":
            name = split.split("_", 1)[1] if split.startswith("ood_") else split
            family = re.sub(r"\d+$", "", name)
            candidates = [
                f"{transcripts_root}/ood/{family}/{split}.tsv",
                f"{transcripts_root}/ood/{family}/{name}.tsv",
                f"{transcripts_root}/ood/{family}/ood_{name}.tsv",
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return None

        path = f"{transcripts_root}/movie10/{split[:-2]}/movie10_{split}.tsv"
        return path if os.path.exists(path) else None

    def _get_transcript_df(self, split: str) -> Optional[pd.DataFrame]:
        if split not in self.transcript_paths:
            return None
        transcript_df = self._transcript_cache.get(split)
        if transcript_df is None:
            transcript_df = pd.read_csv(self.transcript_paths[split], sep="\t")
            self._transcript_cache[split] = transcript_df
        return transcript_df

    def _movie_type_from_split(self, split: str) -> str:
        return _movie_type_from_split_name(split)

    def _stim_split_key(self, split: str) -> str:
        movie_type = self._movie_type_from_split(split)
        if movie_type == "ood" and split.startswith("ood_"):
            return split.split("_", 1)[1]
        return split

    def _get_stim_handle(self, split: str):
        movie_type = self._movie_type_from_split(split)
        pid = os.getpid()
        key = (pid, movie_type)
        handle = self._stim_handles.get(key)
        if handle is None:
            handle = h5py.File(self.stim_paths[movie_type], "r")
            self._stim_handles[key] = handle
        return handle

    def _get_voxel_handle(self, subj: int, split: str):
        movie_type = self._movie_type_from_split(split)
        pid = os.getpid()
        key = (pid, subj, movie_type)
        handle = self._voxel_handles.get(key)
        if handle is None:
            handle = h5py.File(self.voxel_h5_paths[subj][movie_type], "r")
            self._voxel_handles[key] = handle
        return handle

    def _get_voxel_row(self, subj: int, split: str, ind: int) -> np.ndarray:
        handle = self._get_voxel_handle(subj, split)
        voxel_row = np.asarray(handle["voxel"][split][ind], dtype=np.float32)
        if voxel_row.shape[0] != self.valid_voxel_mask.shape[0]:
            raise RuntimeError(
                f"Voxel row size mismatch for split={split}: "
                f"row has {voxel_row.shape[0]} voxels, "
                f"mask expects {self.valid_voxel_mask.shape[0]}."
            )
        return voxel_row[self.valid_voxel_mask]

    def _collect_fmri_targets(self, split: str, ind: int, available_subjects: List[int]):
        fmri_data = {}
        value_key = "parcel" if self.readout_res == "parcels" else "voxel"

        available_arrays = []
        observed_values = {}
        for subj in available_subjects:
            if self.readout_res == "parcels":
                value = self.fmri[subj][split][value_key][ind]
                value = np.asarray(value, dtype=np.float32)
            else:
                value = self._get_voxel_row(subj, split, ind)
            if value is not None:
                observed_values[subj] = value
                available_arrays.append(value)

        if len(available_arrays) == 0:
            raise RuntimeError(f"No available targets for split={split}, ind={ind}")

        fill_value = np.mean(available_arrays, axis=0).astype(np.float32)

        for subj in self.subject_ids:
            if subj in observed_values:
                fmri_data[f"sub_{subj}"] = observed_values[subj]
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
        state["_voxel_handles"] = {}
        return state

    def __del__(self):
        for handle in self._stim_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        for handle in self._voxel_handles.values():
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

        stim_split = self._stim_split_key(split)

        # --- FMRI (explicit/traceable target policy) ---
        if bool(sample.get("has_targets", True)):
            fmri_data = self._collect_fmri_targets(split, ind, available_subjects)
        else:
            fmri_data = {}

        if "text" in self.modality:
            # --- Text (cached transcript table per split) ---
            transcript_df = self._get_transcript_df(split)
            if transcript_df is not None:
                context = extract_language_context(transcript_df, ind)

                text_range = (5, 13) if self.text_backbone in {"metaclip", "openaiclip"} else None

                text_all = extract_text(
                    context,
                    self.text_silence_token,
                    text_range=text_range,
                )
                text_str = " ".join(text_all).strip()
                text_str = text_str if text_str else self.text_silence_token
            else:
                text_str = self.text_silence_token

            # print(text_str)
            text_inputs = self.text_processor.process(text_str)

            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items() }
            data_point["text"] = text_inputs

        if ("video" in self.modality) or ("audio" in self.modality):
            # --- Worker-safe lazy HDF5 handles ---
            stim = self._get_stim_handle(split)

            if "video" in self.modality:
                # --- Video ---
                video_fps = stim[stim_split]["video"].attrs["fps"]
                end = int((ind + 1) * tr * video_fps)
                start = max(int((ind - 14) * tr * video_fps), 0)
                frames_ds = stim[stim_split]["video"]

                if self.sample_hrf:
                    mean, std = -150, 50
                    lower, upper = -669, 0
                    all_frames = np.arange(lower, upper)
                    a, b = (lower - mean) / std, (upper - mean) / std
                    probs = truncnorm.pdf(all_frames, a, b, loc=mean, scale=std)
                    probs /= probs.sum()
                    sample_inds = np.random.choice(all_frames, size=self.num_frames, p=probs, replace=False)
                    video_inputs = self.subsample_frames_lazy(stim, frames_ds, start, end, stim_split, sample_inds)
                else:
                    video_inputs = self.subsample_frames_lazy(stim, frames_ds, start, end, stim_split)

                data_point["video"] = video_inputs

            if "audio" in self.modality:
                # --- Audio ---
                sr = stim[stim_split]["audio"].attrs["sr"]
                end_audio = int((ind + 1) * tr * sr)
                start_audio = max(int((ind - 14) * tr * sr), 0)
                audio = np.asarray(stim[stim_split]["audio"][start_audio:end_audio])
                audio_inputs = self.transform_audio(audio, sr)
                data_point["audio"] = audio_inputs

        return data_point, fmri_data


    def __len__(self):
        return len(self.samples)





