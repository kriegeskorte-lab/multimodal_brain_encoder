BACKBONE_LIST = {
    'text': {
        "llama": "meta-llama/Llama-3.2-1B", 
        "deberta": "microsoft/deberta-v3-large", 
        "metaclip": "facebook/metaclip-2-worldwide-m16"},
    'video': {
        "timesformer": "facebook/timesformer-base-finetuned-k400", 
        # "vivit": "google/vivit-b-16x2-kinetics400", # 32 frames
        "videomae": "OpenGVLab/VideoMAEv2-Base", 
        "dino": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
        "metaclip": "facebook/metaclip-2-worldwide-m16"},
    'audio': {
        "wav2vec2": "facebook/wav2vec2-base-960h", 
        "whisper": "openai/whisper-small"}
}


CACHE_DIR = "/engram/nklab/models/hf_cache"

if __name__ == "__main__":

    import torch
    import av
    import numpy as np
    from transformers import AutoTokenizer, AutoImageProcessor, AutoModel, AutoConfig, AutoProcessor, VideoMAEImageProcessor, Wav2Vec2Processor, WhisperProcessor, TimesformerModel, MetaClip2VisionModel, MetaClip2TextModel, Wav2Vec2Model, WhisperModel
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset, Audio
    from transformers.utils import logging
    logging.set_verbosity_error()
    import librosa
    import h5py
    from pathlib import Path
    import sys

    
    def extract_audio_segment(stim_audio, sr, ind, tr, context=14):
        end_audio = int((ind + 1) * tr * sr)
        start_audio = max(int((ind - context) * tr * sr), 0)
        return stim_audio[start_audio:end_audio]

    def preprocess_audio(audio, sr, target_sr=16000, target_len=492818):
        # resample
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # pad or truncate
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
    audio_samples = [audio_1, audio_2]

    audio_backbone = "wav2vec2"  # "wav2vec2" or "whisper"
    model_name = BACKBONE_LIST['audio'][audio_backbone]
    print(f"Testing {audio_backbone} audio backbone with model: {model_name}")
    if audio_backbone == "whisper":
        audio_processor = WhisperProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=True)
        model = WhisperModel.from_pretrained(
            model_name,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = audio_processor(
            audio_samples,
            sampling_rate=target_sr,
            return_tensors="pt",
        ).to(model.device)
        print("Input waveform shape:", inputs["input_features"].shape)  # torch.Size([2, 80, 3000])  

        with torch.inference_mode():
            outputs = model.encoder(**inputs)

        features = outputs.last_hidden_state.mean(dim=1)  # torch.Size([2, 1500, 768])   -> torch.Size([2, 768])
        print(f"{audio_backbone} features shape:", features.shape) # torch.Size([2, 768]) for whisper-small

    elif audio_backbone == "wav2vec2":
        audio_processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=True)
        model = Wav2Vec2Model.from_pretrained(
            model_name,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = audio_processor(
            audio_samples,
            sampling_rate=target_sr,
            return_tensors="pt",
        ).to(model.device)
        print("Input waveform shape:", inputs["input_values"].shape)  # torch.Size([2, 492818])

        with torch.inference_mode():
            outputs = model(**inputs)

        features = outputs.last_hidden_state.mean(dim=1)  # torch.Size([2, 1539, 768]) -> torch.Size([2, 768])
        print(f"{audio_backbone} features shape:", features.shape) # torch.Size([2, 768]) for wav2vec2-base
    
    print("\n" + "="*50 + "\n")
    texts = [
        "Columbia prepares the next generation of thinkers, scientists, artists, and leaders through an education grounded in intellectual rigor, open inquiry, and a commitment to the public good. ",
        "At Columbia's Zuckerman Institute, we believe that understanding how the brain works — and gives rise to mind and behavior — is the most urgent and exciting challenge of our time."
    ]

    text_backbone = "deberta"  # "llama", "deberta", or "metaclip"
    model_name = BACKBONE_LIST['text'][text_backbone]
    print(f"Testing {text_backbone} text backbone with model: {model_name}")
    if text_backbone == "deberta":
        text_processor = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = text_processor(text=texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(model.device)
        print("Input token ids shape:", inputs['input_ids'].shape)  # torch.Size([2, 37]) 
        with torch.inference_mode():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state # torch.Size([2, 37, 1024])   
        mask = inputs["attention_mask"].unsqueeze(-1) # torch.Size([2, 37]) -> torch.Size([2, 37, 1])
        features = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        print(f"{text_backbone} features shape:", features.shape) # torch.Size([2, 1024])

    elif text_backbone == "llama":
        text_processor = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        # Llama3 has no pad token -> add one
        if text_processor.pad_token is None:
            text_processor.add_special_tokens({"pad_token": "<pad>"})
            # resize embeddings if new token added
            # The pad token embedding will almost never matter because it is masked by the attention mask.
            model.resize_token_embeddings(len(text_processor), mean_resizing=False) 
            model.config.pad_token_id = text_processor.pad_token_id


        inputs = text_processor(text=texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(model.device)
        print("Input token ids shape:", inputs['input_ids'].shape)  # torch.Size([2, 512])
        with torch.inference_mode():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state # torch.Size([2, 37, 1024])   
        mask = inputs["attention_mask"].unsqueeze(-1) # torch.Size([2, 37]) -> torch.Size([2, 37, 1])
        features = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        print(f"{text_backbone} features shape:", features.shape) # torch.Size([2, 2048])

    elif text_backbone == "metaclip":
        text_processor = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=CACHE_DIR)
        model = MetaClip2TextModel.from_pretrained(
            model_name,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = text_processor(text=texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(model.device)
        print("Input token ids shape:", inputs['input_ids'].shape)  # torch.Size([2, 38])
        with torch.inference_mode():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state # torch.Size([2, 77, 512])   
        mask = inputs["attention_mask"].unsqueeze(-1) # torch.Size([2, 77]) -> torch.Size([2, 77, 1])
        features = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        print(f"{text_backbone} features shape:", features.shape) # torch.Size([2, 512])
    
    print("\n" + "="*50 + "\n")

    video_path = hf_hub_download(
        repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset", cache_dir=CACHE_DIR
    )

    def read_video(video_path):
        container = av.open(video_path)
        frames = []

        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

        return np.stack(frames)

    video_1 = list(read_video(video_path)[:16])
    video_2 = list(read_video(video_path)[:16])

    videos = [video_1, video_2]

    B = len(videos)      # 2
    T = len(videos[0])   # 16

    flat_frames = [f for v in videos for f in v]

    vide_backbone = "dino"  # "timesformer", "videomae", "dino", "metaclip"
    model_name = BACKBONE_LIST['video'][vide_backbone]
    if vide_backbone == "metaclip":
        video_processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=True)
        model = MetaClip2VisionModel.from_pretrained(
            model_name,
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = video_processor(images=flat_frames, return_tensors="pt").to(model.device)
        print("Input pixel values shape:", inputs['pixel_values'].shape)  # torch.Size([32, 3, 224, 224])
        with torch.inference_mode():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        D = last_hidden_state.shape[-1]  # 768
        last_hidden_state = last_hidden_state.reshape(B, T, -1, D)  # torch.Size([32, 197, 768]) -> torch.Size([2, 16, 197, 768])
        features = last_hidden_state.mean(dim=(1,2))  # torch.Size([2, 16, 768]) -> torch.Size([2, 768])
        print(f"{vide_backbone} features shape:", features.shape)

    elif vide_backbone == "dino":
        video_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = video_processor(images=flat_frames, return_tensors="pt").to(model.device)
        print("Input pixel values shape:", inputs['pixel_values'].shape)  # torch.Size([32, 3, 224, 224])
        with torch.inference_mode():
            outputs = model(**inputs)

        # pooled_output = outputs.pooler_output.reshape(B, T, -1)  # torch.Size([32, 768]) -> torch.Size([2, 16, 768])
        # features = pooled_output.mean(dim=1)  # torch.Size([2, 16, 768]) -> torch.Size([2, 768])
        # print(f"{backbone} features shape:", features.shape)

        last_hidden_state = outputs.last_hidden_state
        D = last_hidden_state.shape[-1]  # 768
        last_hidden_state = last_hidden_state.reshape(B, T, -1, D)  # torch.Size([32, 197, 768]) -> torch.Size([2, 16, 197, 768])
        features = last_hidden_state.mean(dim=(1,2))  # torch.Size([2, 16, 768]) -> torch.Size([2, 768])
        print(f"{vide_backbone} features shape:", features.shape)

    elif vide_backbone == "videomae":
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        video_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(
            model_name, 
            config=config, 
            trust_remote_code=True, 
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        # prepare video for the model
        inputs = video_processor(videos, return_tensors="pt").to(model.device)
        inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
        print("Input pixel values shape:", inputs['pixel_values'].shape)  # storch.Size([2, 3, 16, 224, 224])

        # forward pass
        with torch.inference_mode():
            outputs = model(**inputs)

        print(f"{vide_backbone} output shape:", outputs.shape) # torch.Size([1, 768])
    elif vide_backbone == "timesformer":
        video_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=False)
        model = TimesformerModel.from_pretrained(
            model_name, 
            use_safetensors=True,
            cache_dir=CACHE_DIR
        ).cuda()

        inputs = video_processor(videos, return_tensors="pt").to(model.device)
        print("Input pixel values shape:", inputs['pixel_values'].shape)  # torch.Size([2, 16, 3, 224, 224])
        with torch.inference_mode():
            outputs = model(**inputs)

        features = outputs.last_hidden_state.mean(dim=1)  # torch.Size([2, 3127, 768]) -> torch.Size([2, 768])
        print(f"{vide_backbone} features shape:", features.shape)