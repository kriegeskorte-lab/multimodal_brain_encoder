from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from unicodedata import name

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    MetaClip2TextModel,
    MetaClip2VisionModel,
    TimesformerModel,
    VideoMAEImageProcessor,
    Wav2Vec2Model,
    Wav2Vec2BertModel,
    Wav2Vec2Processor,
    WhisperModel,
    WhisperProcessor,
    CLIPTextModel,
    CLIPVisionModel,
)
import open_clip


BACKBONE_LIST = {
    "text": {
        "llama": "meta-llama/Llama-3.2-1B",
        "deberta": "microsoft/deberta-v3-base",
        "metaclip": "facebook/metaclip-2-worldwide-l14",
        "openaiclip": "openai/clip-vit-large-patch14"
    },
    "video": {
        "timesformer": "facebook/timesformer-base-finetuned-k400",
        "videomae": "OpenGVLab/VideoMAEv2-Base",
        "dino": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "metaclip": "facebook/metaclip-2-worldwide-b16",
        "openaiclip": "openai/clip-vit-large-patch14"
        # "eupe": "facebook/EUPE-ViT-B",
    },
    "audio": {
        "wav2vec": "facebook/w2v-bert-2.0",
        "whisper": "openai/whisper-small",
    },
}

video2frames_defaults = {
    "dino": 16,
    "metaclip": 16,
    "videomae": 16,
    "timesformer": 16,
}


CACHE_DIR = "/engram/nklab/models/hf_cache"


TensorLikeDict = Dict[str, Any]


class ProcessorWrapper:
    def __init__(
        self,
        modality: str,
        backbone: str,
        cache_dir: str = CACHE_DIR,
    ) -> None:
        self.modality = modality
        self.backbone = backbone
        self.model_name = BACKBONE_LIST[modality][backbone]
        self.cache_dir = cache_dir
        self.processor = self._build_processor()
        self._prepare_processor()

    def _prepare_processor(self):
        if self.modality == "text" and self.backbone == "llama":
            if self.processor.pad_token is None:
                if self.processor.eos_token is not None:
                    self.processor.pad_token = self.processor.eos_token
                else:
                    self.processor.add_special_tokens({"pad_token": "<pad_llama>"})
                # print(f"Added pad token to tokenizer: {self.processor.pad_token}") # <|end_of_text|>

    def _build_processor(self):
        if self.modality == "text":
            if self.backbone == "deberta":
                return AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "llama":
                return AutoTokenizer.from_pretrained(
                    self.model_name,
                    padding_side="left",
                    use_fast=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "metaclip":
                return AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "openaiclip":
                return AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    cache_dir=self.cache_dir,
                )

        if self.modality == "audio":
            if self.backbone == "whisper":
                return WhisperProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                )
            if self.backbone == "wav2vec":
                return AutoFeatureExtractor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                )

        if self.modality == "video":
            if self.backbone == "metaclip":
                return AutoProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                )
            if self.backbone == "dino":
                return AutoImageProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "videomae":
                return VideoMAEImageProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "timesformer":
                return AutoImageProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=False,
                )
            if self.backbone == "openaiclip":
                return AutoProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                )

        raise ValueError(f"Unsupported modality/backbone pair: {self.modality}/{self.backbone}")

    def process(self, data: Any, **kwargs: Any) -> TensorLikeDict:
        if self.modality == "text":
            max_length = 77 if self.backbone in ["metaclip", "openaiclip"] else 128
            return self.processor(
                text=data,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        if self.modality == "audio":
            sampling_rate = kwargs.get("sampling_rate", 16000)
            return self.processor(
                data,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )

        if self.modality == "video":
            if self.backbone in {"metaclip", "dino", "openaiclip"}:
                # print(f"Original video input shape: {len(data)} videos") # 2 videos with 16 frames each
                return self.processor(images=data, return_tensors="pt") # torch.Size([16, 3, 224, 224])
            
            inputs = self.processor(data, return_tensors="pt")
            if self.backbone == "videomae":
                inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
            return inputs

        raise ValueError(f"Unsupported modality: {self.modality}")


class FeatureWrapper(torch.nn.Module):
    def __init__(
        self,
        modality: str,
        backbone: str,
        model_name: Optional[str] = None,
        cache_dir: str = CACHE_DIR,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.backbone = backbone
        self.model_name = model_name or BACKBONE_LIST[modality][backbone]
        self.cache_dir = cache_dir
        self.model = self._build_model()

    def _build_model(self):
        if self.modality == "text":
            if self.backbone == "deberta":
                return AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "llama":
                return AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "metaclip":
                return MetaClip2TextModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "openaiclip":
                return CLIPTextModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )

        if self.modality == "audio":
            if self.backbone == "whisper":
                return WhisperModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "wav2vec":
                return Wav2Vec2BertModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )

        if self.modality == "video":
            if self.backbone == "metaclip":
                return MetaClip2VisionModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "dino":
                return AutoModel.from_pretrained(
                    self.model_name,
                    attn_implementation="sdpa",
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "videomae":
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                return AutoModel.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "timesformer":
                return TimesformerModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )
            if self.backbone == "openaiclip":
                return CLIPVisionModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    cache_dir=self.cache_dir,
                )

        raise ValueError(f"Unsupported modality/backbone pair: {self.modality}/{self.backbone}")

    def prepare_for_processor(self) -> None:
        if self.modality == "text" and self.backbone == "llama":
            processor_wrapper = ProcessorWrapper(
                modality=self.modality,
                backbone=self.backbone,
                cache_dir=self.cache_dir,
            )
            tokenizer = processor_wrapper.processor
            vocab_size = self.model.get_input_embeddings().num_embeddings
            if len(tokenizer) != vocab_size:
                # print(f"Resizing model embeddings from {vocab_size} to {len(tokenizer)} due to tokenizer changes")
                self.model.resize_token_embeddings(
                    len(tokenizer),
                    mean_resizing=False, # can be ignored by attention masking
                )
            if tokenizer.pad_token_id is not None:
                # print(f"Setting model pad_token_id to {tokenizer.pad_token_id}") # 128001
                # print(f"Current model pad_token_id: {self.model.config.pad_token_id}") # None
                self.model.config.pad_token_id = tokenizer.pad_token_id
            del processor_wrapper

    def extract_features(self, inputs: TensorLikeDict, **kwargs: Any) -> Any:
        device = next(self.parameters()).device
        inputs = to_device(inputs, device)

        if self.modality == "video" and self.backbone in {"metaclip", "dino", "openaiclip"}:
            pixel_values = inputs["pixel_values"]
            if pixel_values.ndim == 5:
                batch_size, time_steps, channels, height, width = pixel_values.shape
                inputs["pixel_values"] = pixel_values.reshape(
                    batch_size * time_steps,
                    channels,
                    height,
                    width,
                )

        if self.modality == "video" and self.backbone == "videomae":
            with torch.no_grad():
                hidden_states = self.model.model.patch_embed(inputs["pixel_values"])
                hidden_states = self.model.model.pos_drop(hidden_states)
                for block in self.model.model.blocks:
                    hidden_states = block(hidden_states)
            return hidden_states

        if self.modality == "audio" and self.backbone == "whisper":
            with torch.no_grad():
                outputs = self.model.encoder(**inputs)
        else:
            with torch.no_grad():
                # print(inputs.keys())
                outputs = self.model(**inputs)

        if self.modality == "text":
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return hidden_states * mask

        if self.modality == "audio":
            return outputs.last_hidden_state

        if self.modality == "video" and self.backbone in {"metaclip", "dino", "openaiclip"}:
            batch_size = kwargs.get("batch_size")
            time_steps = kwargs.get("time_steps")
            if batch_size is None or time_steps is None:
                raise ValueError("batch_size and time_steps are required for metaclip/dino/openaiclip video inference")
            hidden_states = outputs.last_hidden_state
            hidden_size = hidden_states.shape[-1]
            return hidden_states.reshape(batch_size, time_steps, -1, hidden_size)

        if self.modality == "video" and self.backbone == "timesformer":
            return outputs.last_hidden_state

        raise ValueError(f"Unsupported modality/backbone pair for feature extraction: {self.modality}/{self.backbone}")


def to_device(batch: TensorLikeDict, device: torch.device) -> TensorLikeDict:
    moved: TensorLikeDict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _flatten_video_frames(videos_or_frames: Any) -> List[Any]:
    if not isinstance(videos_or_frames, Sequence) or len(videos_or_frames) == 0:
        raise ValueError("Expected a non-empty sequence for video input")

    first_item = videos_or_frames[0]
    if isinstance(first_item, Sequence) and not isinstance(first_item, (bytes, str)):
        return [frame for video in videos_or_frames for frame in video]
    return list(videos_or_frames)

def inspect_special_tokens(tokenizer):
    print("=== Special Tokens ===")
    print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("unk_token:", tokenizer.unk_token, tokenizer.unk_token_id)

    # Not all tokenizers have sep
    sep_token = getattr(tokenizer, "sep_token", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    print("sep_token:", sep_token, sep_token_id)

    print("\nAll special tokens:", tokenizer.special_tokens_map)


if __name__ == "__main__":
    import av
    import h5py
    import librosa
    import numpy as np
    from huggingface_hub import hf_hub_download
    from transformers.utils import logging

    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_test(modality: str, backbone: str) -> None:
        print(f"Testing {modality}/{backbone} with model: {BACKBONE_LIST[modality][backbone]}")

        processor_wrapper = ProcessorWrapper(
            modality=modality,
            backbone=backbone,
            cache_dir=CACHE_DIR,
        )
        feature_wrapper = FeatureWrapper(
            modality=modality,
            backbone=backbone,
            cache_dir=CACHE_DIR,
        )
        feature_wrapper.prepare_for_processor()
        feature_wrapper.model = feature_wrapper.model.to(device)

        infer_kwargs: Dict[str, Any] = {}

        if modality == "text":
            samples = [
                "Columbia prepares the next generation of thinkers, scientists, artists, and leaders through an education grounded in intellectual rigor, open inquiry, and a commitment to the public good.",
                "At Columbia's Zuckerman Institute, we believe that understanding how the brain works is an urgent and exciting challenge.",
            ]
            inputs = processor_wrapper.process(samples)

            tokenizer = processor_wrapper.processor
            inspect_special_tokens(tokenizer)

        elif modality == "audio":
            def extract_audio_segment(stim_audio, sr, ind, tr, context=14):
                end_audio = int((ind + 1) * tr * sr)
                start_audio = max(int((ind - context) * tr * sr), 0)
                return stim_audio[start_audio:end_audio]

            def preprocess_audio(audio, sr, target_sr=16000, target_len=492818):
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                if audio.shape[0] < target_len:
                    pad_size = target_len - audio.shape[0]
                    audio = np.pad(audio, (0, pad_size), mode="constant")
                else:
                    audio = audio[:target_len]
                return audio

            stim_path = "/engram/nklab/datasets/algonauts_2025.competitors/stimuli/movies/friends_smaller.h5"
            split = "s01e01a"
            tr = 1.49
            target_sr = 16000

            with h5py.File(stim_path, "r") as stim:
                sr = stim[split]["audio"].attrs["sr"]
                audio_data = stim[split]["audio"]
                audio_1 = extract_audio_segment(audio_data, sr, ind=20, tr=tr)
                audio_2 = extract_audio_segment(audio_data, sr, ind=50, tr=tr)

            audio_1 = preprocess_audio(audio_1, sr)
            audio_2 = preprocess_audio(audio_2, sr)
            samples = [audio_1, audio_2]
            inputs = processor_wrapper.process(samples, sampling_rate=target_sr)

        elif modality == "video":
            video_path = hf_hub_download(
                repo_id="nielsr/video-demo",
                filename="eating_spaghetti.mp4",
                repo_type="dataset",
                cache_dir=CACHE_DIR,
            )

            def read_video(path):
                container = av.open(path)
                frames = []
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
                return np.stack(frames)

            video_1 = list(read_video(video_path)[:16])
            video_2 = list(read_video(video_path)[:16])
            videos = [video_1, video_2]
            # videos = [video_1] # a list of videos, where each video is a list of frames (HWC numpy arrays)

            batch_size = len(videos)
            time_steps = len(videos[0])
            inputs = processor_wrapper.process(videos) 
            # print("mbb test:", inputs['pixel_values'].shape) # torch.Size([32, 3, 224, 224]) after flattening frames in processor
            if backbone in {"dino", "metaclip", "openaiclip"}:
                infer_kwargs["batch_size"] = batch_size
                infer_kwargs["time_steps"] = time_steps

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # print(inputs['pixel_values'].shape) 
        inputs = to_device(inputs, device)
        features = feature_wrapper.extract_features(inputs, **infer_kwargs)

        print("Input keys:", list(inputs.keys()))
        if isinstance(features, torch.Tensor):
            print("Feature shape:", features.shape)
        elif hasattr(features, "shape"):
            print("Output shape:", features.shape)
        else:
            print("Output type:", type(features))
        print("=" * 50)

    # Toggle these for quick checks.
    tests = [
        ("text", "deberta"),
        ("text", "llama"),
        ("text", "metaclip"),
        ("text", "openaiclip"),
        # ("audio", "wav2vec"),
        # ("audio", "whisper"),
        # ("video", "videomae"),
        # ("video", "timesformer"),
        # ("video", "dino"),
        # ("video", "metaclip"),
        # ("video", "openaiclip"),
    ]

    for modality_name, backbone_name in tests:
        run_test(modality_name, backbone_name)
        # try:
        #     run_test(modality_name, backbone_name)
        # except Exception as exc:
        #     print(f"Failed {modality_name}/{backbone_name}: {exc}")
        #     print("=" * 50)


'''
Testing text/deberta with model: microsoft/deberta-v3-large                   
Input keys: ['input_ids', 'token_type_ids', 'attention_mask']                 
Feature shape: torch.Size([2, 512, 1024])                                     
==================================================                            
Testing text/llama with model: meta-llama/Llama-3.2-1B                        
Input keys: ['input_ids', 'attention_mask']                                   
Feature shape: torch.Size([2, 512, 2048])                                     
==================================================                            
Testing text/metaclip with model: facebook/metaclip-2-worldwide-m16           
Input keys: ['input_ids', 'attention_mask']                                   
Feature shape: torch.Size([2, 77, 512])                                       
==================================================                            
Testing audio/wav2vec with model: facebook/w2v-bert-2.0               
Input keys: ['input_features', 'attention_mask']                                                 
Feature shape: torch.Size([2, 1539, 1024])                                    
==================================================                            
Testing audio/whisper with model: openai/whisper-small                        
Input keys: ['input_features']                                                
Feature shape: torch.Size([2, 1500, 768])                                     
==================================================                            
Testing video/videomae with model: OpenGVLab/VideoMAEv2-Base                  
Input keys: ['pixel_values']                                                  
Feature shape: torch.Size([2, 1568, 768]) # 1568 = 8 temporal tokens * 14 * 14                                      
==================================================                            
Testing video/timesformer with model: facebook/timesformer-base-finetuned-k400
Input keys: ['pixel_values']                                                  
Feature shape: torch.Size([2, 3137, 768]) # 3137 = 16 frames * 14 * 14 + 1 (cls token)                               
==================================================                            
Testing video/dino with model: facebook/dinov3-vitb16-pretrain-lvd1689m 
Input keys: ['pixel_values']              
Feature shape: torch.Size([2, 16, 201, 768])  # 201 = 14 * 14 + 1 (cls token) + 4 (registers)                               
==================================================                            
Testing video/metaclip with model: facebook/metaclip-2-worldwide-m16          
Input keys: ['pixel_values']                                                  
Feature shape: torch.Size([2, 16, 197, 512])  # 197 = 14 * 14 + 1 (cls token)                                    
================================================== 
Testing video/openaiclip with model: openai/clip-vit-large-patch14
Input keys: ['pixel_values']
Feature shape: torch.Size([2, 16, 257, 1024]) # 257 = 16 * 16 + 1 (cls tokens)
==================================================
Testing text/openaiclip with model: openai/clip-vit-large-patch14
Input keys: ['input_ids', 'attention_mask']
Feature shape: torch.Size([2, 77, 768])
==================================================
'''