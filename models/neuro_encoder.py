import random
import math
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace

import torch
from torch import nn


import sys
sys.path.append("/engram/nklab/pf2477")

from multimodal_encoder.args import get_args_parser
from multimodal_encoder.cneuro_dataset.cneuro_data import algonauts_dataset
from multimodal_encoder.utils.utils import NestedTensor
from multimodal_encoder.models.transformer import build_transformer
from multimodal_encoder.models.multimodel_backbone import BACKBONE_LIST, ProcessorWrapper, FeatureWrapper, to_device

'''
Sensor: Receives multimodal inputs and projects them into a shared token space.
PerceptualAligner: Aligns projected multimodal tokens using standard transformer.
Readout: Readout module that predicts fMRI from aligned token representations.
'''

CACHE_DIR = "/engram/nklab/models/hf_cache"

class Sensor(nn.Module):
    """Receives multimodal inputs and projects them into a shared token space."""

    def __init__(self, args):
        super().__init__()

        self.modality = args.modality
        self.d_model = args.hidden_dim
        self.video_backbone = getattr(args, "video_backbone", "metaclip").lower()
        self.text_backbone = getattr(args, "text_backbone", "metaclip").lower()
        self.audio_backbone = getattr(args, "audio_backbone", "whisper").lower()

        if "video" in self.modality:
            self.video_model = FeatureWrapper(
                modality="video",
                backbone=self.video_backbone,
                cache_dir=CACHE_DIR,
            )
            for p in self.video_model.model.parameters():
                p.requires_grad = False
            self.video_proj = nn.LazyLinear(self.d_model)
            self.video_layernorm = nn.LayerNorm(self.d_model)
            
        if "audio" in self.modality:
            self.audio_model = FeatureWrapper(
                modality="audio",
                backbone=self.audio_backbone,
                cache_dir=CACHE_DIR,
            )
            for p in self.audio_model.model.parameters():
                p.requires_grad = False
            self.audio_proj = nn.LazyLinear(self.d_model)
            self.audio_layernorm = nn.LayerNorm(self.d_model)
            
        if "text" in self.modality: 
            self.text_model = FeatureWrapper(
                modality="text",
                backbone=self.text_backbone,
                cache_dir=CACHE_DIR,
            )
            for p in self.text_model.model.parameters():
                p.requires_grad = False
            self.text_proj = nn.LazyLinear(self.d_model)
            self.text_layernorm = nn.LayerNorm(self.d_model)

        # LayerNorm can help stabilize training in multimodal fusion.
        self.multimodal_projector = nn.ModuleDict()
        if "video" in self.modality:
            self.multimodal_projector["video"] = nn.Sequential(self.video_proj, self.video_layernorm)
        if "audio" in self.modality:
            self.multimodal_projector["audio"] = nn.Sequential(self.audio_proj, self.audio_layernorm)
        if "text" in self.modality:
            self.multimodal_projector["text"] = nn.Sequential(self.text_proj, self.text_layernorm)
    
    def train(self, mode: bool = True):
        super().train(mode)
        # Keep pretrained feature backbones in eval mode even when parent model is training.
        if hasattr(self, "video_model"):
            self.video_model.model.eval()
        if hasattr(self, "audio_model"):
            self.audio_model.model.eval()
        if hasattr(self, "text_model"):
            self.text_model.model.eval()
        return self

    def _encode_video(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["video"]
        inputs = to_device(inputs, device)

        infer_kwargs: Dict[str, Any] = {}
        infer_kwargs["batch_size"] = inputs["pixel_values"].shape[0]
        infer_kwargs["time_steps"] = inputs["pixel_values"].shape[1]
        with torch.no_grad():
            video_feat = self.video_model.extract_features(inputs, **infer_kwargs)  # [B, C]
        return self.multimodal_projector["video"](video_feat)

    def _encode_audio(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["audio"]
        inputs = to_device(inputs, device)
        with torch.no_grad():
            audio_feat = self.audio_model.extract_features(inputs)  # [B, C]
        return self.multimodal_projector["audio"](audio_feat)

    def _encode_text(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["text"]
        inputs = to_device(inputs, device)
        with torch.no_grad():
            text_feat = self.text_model.extract_features(inputs)  # [B, C]
        return self.multimodal_projector["text"](text_feat)

    def forward(self, samples: NestedTensor) -> torch.Tensor:
        device = next(self.parameters()).device
        modality_tokens = {
            "video": None,
            "audio": None,
            "text": None,
        }

        if "video" in self.modality:
            modality_tokens["video"] = self._encode_video(samples, device)

        if "audio" in self.modality:
            modality_tokens["audio"] = self._encode_audio(samples, device)

        if "text" in self.modality:
            modality_tokens["text"] = self._encode_text(samples, device)

        return modality_tokens


class PerceptualAligner(nn.Module):
    """Aligns projected multimodal tokens using standard/custom DETR-style transformer."""

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.modality = args.modality
        self.d_model = args.hidden_dim
        self.output_norm = nn.LayerNorm(self.d_model)
        self.transformer = build_transformer(args)
        self.modality_dropout_prob = float(getattr(args, "modality_dropout", 0.2))
        self.query_embed = nn.Embedding(args.num_queries, self.d_model) # learned positional embedding for query slots.
        self._build_pos_embedding()

    def _build_pos_embedding(self):
        self.model2sequence_length = {
            "wav2vec": 1539,
            "whisper": 1500,
            "deberta": 512,
            "llama": 512,
            "metacliptext": 77,
            "metaclipvision": 197,
            "timesformer": 3137,
            "videomae": 1568,
            "dino": 201,
            "openaiclipvision": 257,
            "openaicliptext": 77,
        }
        if "audio" in self.modality:
            self.audio_backbone = self.args.audio_backbone.lower()
            num_audio_tokens = self.model2sequence_length[self.audio_backbone]
            self.audio_pos_embedding = nn.Embedding(num_audio_tokens, self.d_model)
        if "text" in self.modality: 
            self.text_backbone = self.args.text_backbone.lower()
            if self.text_backbone == 'metaclip':
                num_text_tokens = self.model2sequence_length["metacliptext"]
            elif self.text_backbone == 'openaiclip':
                num_text_tokens = self.model2sequence_length["openaicliptext"]
            else:
                num_text_tokens = self.model2sequence_length[self.text_backbone]
            self.text_pos_embedding = nn.Embedding(num_text_tokens, self.d_model)
        if "video" in self.modality:
            self.video_backbone = self.args.video_backbone.lower()
            # Current models all use 14x14 = 196 patch tokens per temporal step/frame.

            if self.video_backbone == "videomae":
                # 16 frames with tubelet_size=2 -> 8 temporal tokens
                self.video_num_spatial = 196
                self.video_num_temporal = 8
                self.video_num_special = 0

            elif self.video_backbone == "timesformer":
                # 16 frames, flattened later; 1 CLS token total
                self.video_num_spatial = 196
                self.video_num_temporal = 16
                self.video_num_special = 1

            elif self.video_backbone == "dino":
                # 16 frames, per frame: 1 CLS + 4 register + 196 patch tokens
                self.video_num_spatial = 196
                self.video_num_temporal = 16
                self.video_num_special = 5

            elif self.video_backbone == "metaclip":
                # 16 frames, per frame: 1 CLS + 196 patch tokens
                self.video_num_spatial = 196
                self.video_num_temporal = 16
                self.video_num_special = 1

            elif self.video_backbone == "openaiclip":
                # 16 frames, per frame: 1 CLS + 256 patch tokens
                self.video_num_spatial = 256
                self.video_num_temporal = 16
                self.video_num_special = 1
            else:
                raise ValueError(f"Unsupported video backbone: {self.video_backbone}")

            self.video_temporal_embed = nn.Embedding(self.video_num_temporal, self.d_model)
            self.video_spatial_embed = nn.Embedding(self.video_num_spatial, self.d_model)

            if self.video_num_special > 0:
                self.video_special_embed = nn.Embedding(self.video_num_special, self.d_model)
            else:
                self.video_special_embed = None

            self.get_video_pos_encoding = self._build_video_pos_embedding()

    def _build_video_pos_embedding(self):
        """
        Build video positional embeddings matching the shape of video_tokens.

        Inputs by backbone:
            videomae:   [B, 1568, D]        = [B, 8 * 196, D]
            timesformer:[B, 3137, D]        = [B, 1 + 16 * 196, D]
            dino:       [B, 16, 201, D]     = [B, 16, 5 + 196, D]
            metaclip:   [B, 16, 197, D]     = [B, 16, 1 + 196, D]
            openaiclip: [B, 16, 257, D]     = [B, 16, 1 + 256, D]
        Outputs: 
            pos_embed:  [1568, D]        = [8 * 196, D]
            pos_embed:  [3137, D]        = [1 + 16 * 196, D]
            pos_embed:  [16, 201, D]     = [16, 5 + 196, D]
            pos_embed:  [16, 197, D]     = [16, 1 + 196, D]
            pos_embed:  [16, 257, D]     = [16, 1 + 256, D]
        """
        def build() -> torch.Tensor:

            if self.video_backbone == "videomae":
                temp = self.video_temporal_embed.weight[:, None, :]  # [T,1,D]
                spat = self.video_spatial_embed.weight[None, :, :]    # [1,S,D]
                pos = (temp + spat).reshape(self.video_num_temporal * self.video_num_spatial, self.d_model)
                return pos  # [1568, D]

            if self.video_backbone == "timesformer":
                temp = self.video_temporal_embed.weight[:, None, :]  # [T,1,D]
                spat = self.video_spatial_embed.weight[None, :, :]    # [1,S,D]
                patch = (temp + spat).reshape(self.video_num_temporal * self.video_num_spatial, self.d_model)  # [3136, D]
                cls = self.video_special_embed.weight  # [1, D]
                return torch.cat([cls, patch], dim=0)  # [3137, D]

            if self.video_backbone == "dino":
                temp = self.video_temporal_embed.weight[:, None, :]  # [T,1,D]
                special = self.video_special_embed.weight  # [5, D]
                spatial = self.video_spatial_embed.weight  # [196, D]
                frame = torch.cat([special, spatial], dim=0)  # [201, D]
                return frame.unsqueeze(0) + temp  # [16, 201, D]

            if self.video_backbone == "metaclip":
                temp = self.video_temporal_embed.weight[:, None, :]  # [T,1,D]
                special = self.video_special_embed.weight  # [1, D]
                spatial = self.video_spatial_embed.weight  # [196, D]
                frame = torch.cat([special, spatial], dim=0)  # [197, D]
                return frame.unsqueeze(0) + temp  # [16, 197, D]
            
            if self.video_backbone == "openaiclip":
                temp = self.video_temporal_embed.weight[:, None, :]  # [T,1,D]
                special = self.video_special_embed.weight  # [1, D]
                spatial = self.video_spatial_embed.weight  # [256, D]
                frame = torch.cat([special, spatial], dim=0)  # [257, D]
                return frame.unsqueeze(0) + temp  # [16, 257, D]

            raise ValueError(f"Unsupported video backbone: {self.video_backbone}")

        return build

    def modality_dropout(self, video, audio, text, p):
        keep_map = {
            "video": None if video is None else None,
            "audio": None if audio is None else None,
            "text": None if text is None else None,
        }
        if not self.training or p <= 0:
            if video is not None:
                keep_map["video"] = torch.ones(video.shape[0], device=video.device, dtype=torch.bool)
            if audio is not None:
                keep_map["audio"] = torch.ones(audio.shape[0], device=audio.device, dtype=torch.bool)
            if text is not None:
                keep_map["text"] = torch.ones(text.shape[0], device=text.device, dtype=torch.bool)
            return video, audio, text, keep_map

        modalities = [video, audio, text]
        present = [i for i, m in enumerate(modalities) if m is not None]
        assert len(present) > 0, "At least one modality must be present"

        B = modalities[present[0]].shape[0]
        device = modalities[present[0]].device
        keep = torch.rand(B, len(present), device=device) > p  # per-sample, per-present-modality

        # ensure at least one modality survives per sample
        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            idx = all_dropped.nonzero(as_tuple=False).squeeze(1)
            rescue = torch.randint(0, len(present), (idx.numel(),), device=device)
            keep[idx, rescue] = True

        for col, m_idx in enumerate(present):
            m = modalities[m_idx]
            if m is not None:
                scale_shape = [B] + [1] * (m.ndim - 1)
                modalities[m_idx] = m * keep[:, col].view(*scale_shape)
                if m_idx == 0:
                    keep_map["video"] = keep[:, col]
                elif m_idx == 1:
                    keep_map["audio"] = keep[:, col]
                else:
                    keep_map["text"] = keep[:, col]

        return modalities[0], modalities[1], modalities[2], keep_map
    
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # x: dict of modality -> tokens; video may be [B, F, S, D]

        video_x = x.get("video", None)
        audio_x = x.get("audio", None)
        text_x  = x.get("text", None)

        # add positional encodings per modality
        pos_embed = []
        if video_x is not None:
            video_pos = self.get_video_pos_encoding()  # [S, D] or [F, S, D] depending on backbone
            # video_x = video_x + video_pos.unsqueeze(0)
            if video_x.ndim == 4:
                Bv, F, S, Dv = video_x.shape
                video_x = video_x.reshape(Bv, F * S, Dv)
            if video_pos.ndim == 3:
                F, S, D = video_pos.shape
                video_pos = video_pos.reshape(F * S, D)
            pos_embed.append(video_pos)

        if audio_x is not None:
            audio_pos = self.audio_pos_embedding.weight  # [Ta, D]
            # audio_x = audio_x + audio_pos.unsqueeze(0)
            pos_embed.append(audio_pos)

        if text_x is not None:
            text_pos = self.text_pos_embedding.weight    # [Tt, D]
            # text_x = text_x + text_pos.unsqueeze(0)
            pos_embed.append(text_pos)

        pos_embed = torch.cat(pos_embed, dim=0)  # [Tv+Ta+Tt, D]

        # modality dropout after positional encodings (per sample, whole modality)
        video_x, audio_x, text_x, keep_map = self.modality_dropout(
            video_x, audio_x, text_x, self.modality_dropout_prob
        )

        tokens = []
        src_masks = []

        if video_x is not None:
            Bv = video_x.shape[0]
            tokens.append(video_x)
            video_len = video_x.shape[1]
            video_keep = keep_map["video"]
            video_mask = (~video_keep).unsqueeze(1).expand(Bv, video_len)
            src_masks.append(video_mask)

        if audio_x is not None:
            Ba = audio_x.shape[0]
            tokens.append(audio_x)
            audio_len = audio_x.shape[1]
            audio_keep = keep_map["audio"]
            audio_mask = (~audio_keep).unsqueeze(1).expand(Ba, audio_len)
            src_masks.append(audio_mask)

        if text_x is not None:
            Bt = text_x.shape[0]
            tokens.append(text_x)
            text_len = text_x.shape[1]
            text_keep = keep_map["text"]
            text_mask = (~text_keep).unsqueeze(1).expand(Bt, text_len)
            src_masks.append(text_mask)

        if len(tokens) == 0:
            raise ValueError("No modalities available after dropout.")

        x_cat = torch.cat(tokens, dim=1)  # [B, T, D]
        B, T, _ = x_cat.shape

        # Adapt [B, T, D] to DETR-like interface expected by existing transformer.
        src = x_cat.transpose(1, 2).unsqueeze(-1)  # [B, D, T, 1] content

        # [T, D] -> [1, D, T] -> [1, D, T, 1] 
        pos_embed = pos_embed.unsqueeze(0).transpose(1, 2).unsqueeze(-1) # .repeat(B, 1, 1, 1)  # [B, D, T, 1]
        # src = src + pos_embed  # this makes K/V have positional info, while queries remain purely learnable;
        # pos_embed = None

        # attention padding mask
        src_mask = torch.cat(src_masks, dim=1).to(device=x_cat.device, dtype=torch.bool)  # [B, T]

        # learnable query embeddings, expected shape [num_queries, D]
        query_embed = self.query_embed.weight    # [num_queries, D]

        hidden_states, attn_maps = self.transformer(
            src=src,
            mask=src_mask, 
            query_embed=query_embed,
            pos_embed=pos_embed, # pos_embed is added to src tokens above;
            masks=False # feature-fusion flag (boolean)
        )

        # Expected [L, B, Q, D] if return_intermediate_dec=True.
        outputs = hidden_states[-1] if hidden_states.dim() == 4 else hidden_states
        return self.output_norm(outputs), attn_maps
    

class Readout(nn.Module):
    """Readout module that predicts fMRI from aligned token representations."""

    def __init__(self, args, d_model: int, fmri_out_dim: int):
        super().__init__()
        self.fmri_out_dim = fmri_out_dim
        self.readout_fmri = args.readout_res

        if self.readout_fmri == 'parcels':
            assert args.num_queries == args.num_parcels, \
                "parcels readout requires num_queries == num_parcels"
            self.output_dim = args.num_parcels
            self.readout_head = nn.Linear(d_model, self.output_dim)

        elif self.readout_fmri == 'voxels':
            masked_parcellation = args.masked_parcellation # (172218,) 0 - 1000
            valid_voxel_mask = args.valid_voxel_mask # (172218,) Bool

            if len(masked_parcellation) != len(valid_voxel_mask):
                raise ValueError(
                    "masked_parcellation and valid_voxel_mask must have the same length."
                )
            
            valid_masked_parcellation = masked_parcellation[valid_voxel_mask] # (122721,) 
            valid_labels = torch.as_tensor(valid_masked_parcellation, dtype=torch.long) # torch.Size([122721])
            voxel_to_query = valid_labels - 1  # 1..Q -> 0..Q-1

            self.output_dim = int(valid_labels.numel()) # 1: 122721; 2: 119282

            self.register_buffer("voxel_to_query", voxel_to_query) # torch.Size([122721])
            self.readout_head = nn.Linear(d_model, self.output_dim)

        else:
            raise ValueError(f"Unsupported readout_res: {self.readout_fmri}")

    def forward(self, aligned_tokens: torch.Tensor):
        if self.readout_fmri == 'parcels':
            # aligned_tokens: [B, Q, D], with Q expected to match fmri_out_dim.

            # Efficient equivalent of taking the diagonal after Linear(D -> Q):
            # fmri_pred[b, q] = dot(aligned_tokens[b, q, :], W[q, :]) + b[q]
            weight = self.readout_head.weight  # (Q, D)
            bias = self.readout_head.bias  # (Q, )
            if aligned_tokens.shape[1] != weight.shape[0]:
                raise ValueError(
                    f"Readout mismatch: num_queries={aligned_tokens.shape[1]} "
                    f"but out_features={weight.shape[0]}"
                )
            fmri_pred = (aligned_tokens * weight.unsqueeze(0)).sum(dim=-1)
            if bias is not None:
                fmri_pred = fmri_pred + bias.unsqueeze(0)
        
        elif self.readout_fmri == 'voxels':
            # For every voxel, pick the query token assigned to that voxel.
            x_sel = aligned_tokens[:, self.voxel_to_query, :]   # [B, V, D] [B, 122721, D]
            weight = self.readout_head.weight                   # [V, D]
            bias = self.readout_head.bias                       # [V]

            fmri_pred = (x_sel * weight.unsqueeze(0)).sum(dim=-1)
            if bias is not None:
                fmri_pred = fmri_pred + bias.unsqueeze(0)
        
        # l2_reg = torch.tensor(0.0, device=fmri_pred.device)
        # for p in self.readout_head.parameters():
        #     l2_reg = l2_reg + torch.norm(p)
        l2_reg = None

        return fmri_pred, l2_reg


class NeuroEncoder(nn.Module):
    """
    Sensor -> PerceptualAligner -> Readout.
    """

    def __init__(self, args):
        super().__init__()

        self.modality = args.modality
        self.readout_res = args.readout_res

        # Shared latent dimension
        self.d_model = args.hidden_dim

        # Final output dimensionality (main.py sets num_queries from readout_res)
        self.fmri_out_dim = args.num_queries
        self.sensor = Sensor(args)
        self.perceptual_aligner = PerceptualAligner(args)
        self.readout = Readout(args, self.d_model, self.fmri_out_dim)

    def forward(self, samples: NestedTensor):
        multimodal_tokens = self.sensor(samples)
        multimodal_latents, attn_maps = self.perceptual_aligner(multimodal_tokens)
        fmri_pred, l2_reg = self.readout(multimodal_latents)

        return {
            "fmri_pred": fmri_pred,
            "output_tokens": multimodal_latents,
            "l2_reg": l2_reg,
            "attn_maps": attn_maps
        }


if __name__ == "__main__":
    # Lightweight shape tests using pseudo tokens (no backbone downloads).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = get_args_parser()
    args = parser.parse_args()
    args.test_splits = ["wolf"]
    args.backbone_list = BACKBONE_LIST
    args.text_backbone = random.choice(list(BACKBONE_LIST['text'].keys()))
    args.video_backbone = random.choice(list(BACKBONE_LIST['video'].keys()))
    args.audio_backbone = random.choice(list(BACKBONE_LIST['audio'].keys()))
    # args.text_backbone = "openaiclip"
    # args.video_backbone = "openaiclip"
    # args.audio_backbone = "whisper"
    # args.text_backbone = "metaclip"
    # args.video_backbone = "metaclip"
    # args.audio_backbone = "whisper"
    print(f"video_backbone: {args.video_backbone}, text_backbone: {args.text_backbone}, audio_backbone: {args.audio_backbone}")
    test_dataset = algonauts_dataset(args, include_splits=args.test_splits)

    # args.readout_res = "voxels"
    # args.valid_voxel_mask = test_dataset.valid_voxel_mask if args.readout_res == "voxels" else None
    # args.masked_parcellation = test_dataset.masked_parcellation if args.readout_res == "voxels" else None

    B, D = 16, args.hidden_dim
    common = {
		"batch_size": B,
		"num_workers": 0,
		"pin_memory": True,
		# "persistent_workers": args.num_workers > 0,
		"persistent_workers": False,
		"prefetch_factor": None,  # default is 2, 
	}
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)

    sensor = Sensor(args).to(device)
    sensor.eval()  
    for samples, targets in test_loader:
        # print(samples.keys())
        multimodal_tokens = sensor(samples)
        for modality, tokens in multimodal_tokens.items():
            # print(f"{modality} tokens shape: {tokens.shape}")
            pass
        break  # just one batch for a quick check
    '''
    video tokens shape: torch.Size([2, 16, 197, 256]) 
    audio tokens shape: torch.Size([2, 1500, 256])    
    text tokens shape: torch.Size([2, 77, 256])
    '''

    # x = torch.randn(B, T, D, device=device)
    x = multimodal_tokens

    aligner = PerceptualAligner(args).to(device)
    readout = Readout(args, d_model=D, fmri_out_dim=args.num_queries).to(device)

    # Eval pass (deterministic shape check)
    aligner.eval()
    readout.eval()
    with torch.no_grad():
        aligned, _ = aligner(x)
        fmri_pred, _ = readout(aligned)

    print("=== Shape check (eval) ===")
    print(f"Aligned tokens: {aligned.shape} (expected [B, Q, D])")
    print(f"fMRI prediction: {fmri_pred.shape} (expected [B, Q])")

    assert aligned.shape == (B, args.num_queries, D), "PerceptualAligner output shape mismatch"
    expected_dim = args.num_voxels if args.readout_res == "voxels" else args.num_queries
    assert fmri_pred.shape == (B, expected_dim), "Readout output shape mismatch"

    # Train pass (dropout active)
    aligner.train()
    readout.train()
    aligned_train, _ = aligner(x)
    fmri_pred_train, _ = readout(aligned_train)

    print("=== Shape check (train; modality dropout active) ===")
    print(f"Aligned tokens (train): {aligned_train.shape}")
    print(f"fMRI prediction (train): {fmri_pred_train.shape}")

    assert torch.isfinite(aligned_train).all(), "NaN/Inf detected in aligned_train"
    assert torch.isfinite(fmri_pred_train).all(), "NaN/Inf detected in fmri_pred_train"

    print("All pseudo-data shape checks passed.")
    

