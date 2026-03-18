import random
import math
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace

import torch
from torch import nn


import sys
sys.path.append("/engram/nklab/pf2477")

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

    def _encode_video(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["video"]
        inputs = to_device(inputs, device)

        infer_kwargs: Dict[str, Any] = {}
        infer_kwargs["batch_size"] = inputs["pixel_values"].shape[0]
        infer_kwargs["time_steps"] = inputs["pixel_values"].shape[1]
        video_feat = self.video_model.extract_features(inputs, **infer_kwargs)  # [B, C]
        return self.multimodal_projector["video"](video_feat)

    def _encode_audio(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["audio"]
        inputs = to_device(inputs, device)
        audio_feat = self.audio_model.extract_features(inputs)  # [B, C]
        return self.multimodal_projector["audio"](audio_feat)

    def _encode_text(self, samples: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        inputs = samples["text"]
        inputs = to_device(inputs, device)
        text_feat = self.text_model.extract_features(inputs)  # [B, C]
        return self.multimodal_projector["text"](text_feat)

    def forward(self, samples: NestedTensor) -> torch.Tensor:
        device = next(self.parameters()).device
        modality_tokens: List[torch.Tensor] = []

        if "video" in self.modality:
            modality_tokens.append(self._encode_video(samples, device))

        if "audio" in self.modality:
            modality_tokens.append(self._encode_audio(samples, device))

        if "text" in self.modality:
            modality_tokens.append(self._encode_text(samples, device))

        if len(modality_tokens) == 0:
            raise ValueError("No enabled modality tokens found. Check args.modality.")

        return torch.stack(modality_tokens, dim=1)  # [B, num_modalities, d_model]


class PerceptualAligner(nn.Module):
    """Aligns projected multimodal tokens using standard/custom DETR-style transformer."""

    def __init__(self, args):
        super().__init__()

        self.d_model = args.hidden_dim
        self.output_norm = nn.LayerNorm(self.d_model)
        self.transformer = build_transformer(args)
        self.modality_dropout_prob = float(getattr(args, "modality_dropout", 0.2))
        self.modality_embed = nn.Embedding(len(args.modality), self.d_model) # add modality position info to keys (memory) inside transformer
        self.query_embed = nn.Embedding(args.num_queries, self.d_model) # learned positional embedding for query slots.

    def _modality_dropout(self, x: torch.Tensor, dropout_prob: float) -> torch.Tensor:
        # Modality token dropout on [B, T, D]. Drops whole modalities per sample.
        if (not self.training) or dropout_prob <= 0.0:
            return x

        B, T, _ = x.shape
        token_keep = torch.rand(B, T, device=x.device) > dropout_prob

        all_dropped = ~token_keep.any(dim=1)
        if all_dropped.any():
            dropped_rows = all_dropped.nonzero(as_tuple=False).squeeze(1)
            rescue_cols = torch.randint(T, (dropped_rows.numel(),), device=x.device)
            token_keep[dropped_rows, rescue_cols] = True

        return x.masked_fill(~token_keep.unsqueeze(-1), 0.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = self._modality_dropout(x, self.modality_dropout_prob)
        B, T, _ = x.shape
        # print(x)

        # Adapt [B, T, D] to DETR-like interface expected by existing transformer.
        src = x.transpose(1, 2).unsqueeze(-1)  # [B, D, T, 1] content

        # [B, D] -> [1, B, D] -> [1, B, D, 1] -> [B, D, T, 1], added to keys inside transformer (memory + pos)
        pos_embed = self.modality_embed.weight.unsqueeze(0).transpose(1, 2).unsqueeze(-1).repeat(B, 1, 1, 1)  # [B, D, T, 1]

        # learnable query embeddings, expected shape [num_queries, D]
        query_embed = self.query_embed.weight    # [num_queries, D]

        # attention padding mask
        src_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)  # [B, T]

        hidden_states = self.transformer(
            src=src,
            mask=src_mask, 
            query_embed=query_embed,
            pos_embed=pos_embed,
            masks=False # feature-fusion flag (boolean)
        )

        # Expected [L, B, Q, D] if return_intermediate_dec=True.
        outputs = hidden_states[-1] if hidden_states.dim() == 4 else hidden_states
        return self.output_norm(outputs)

class Readout(nn.Module):
    """Readout module that predicts fMRI from aligned token representations."""

    def __init__(self, args, d_model: int, fmri_out_dim: int):
        super().__init__()
        self.fmri_out_dim = fmri_out_dim
        # Shared projection applied independently to each query token: D -> 1.
        self.readout_fmri = args.readout_res
        self.readout_head = nn.Linear(d_model, 1) # we need to change this if we want to predict voxels

    def forward(self, aligned_tokens: torch.Tensor):
        if self.readout_fmri == 'parcels':
            assert aligned_tokens.shape[1] == self.fmri_out_dim, f"Expected number of query tokens {aligned_tokens.shape[1]} to match fmri_out_dim {self.fmri_out_dim} for parcel readout."
        
        # aligned_tokens: [B, Q, D], with Q expected to match fmri_out_dim.
        fmri_pred = self.readout_head(aligned_tokens).squeeze(-1)  # [B, Q]
        
        l2_reg = torch.tensor(0.0, device=fmri_pred.device)
        for p in self.readout_head.parameters():
            l2_reg = l2_reg + torch.norm(p)

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
        multimodal_latents = self.perceptual_aligner(multimodal_tokens)
        fmri_pred, l2_reg = self.readout(multimodal_latents)

        return {
            "fmri_pred": fmri_pred,
            "output_tokens": multimodal_latents,
            "l2_reg": l2_reg,
        }


if __name__ == "__main__":
    # Lightweight shape tests using pseudo tokens (no backbone downloads).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = SimpleNamespace(
        # Model dims
        hidden_dim=256,
        num_queries=1000,
        modality=["video", "audio", "text"],
        modality_dropout=0.2,
        readout_res="parcels",
        dim_feedforward=1024,
        # Transformer args used by build_transformer
        dropout=0.1,
        nheads=8,
        enc_layers=0,
        dec_layers=2,
        pre_norm=True,
        enc_output_layer=-1,
        # Backbones (won't be used in this test, but required to initialize Sensor/Aligner)
        text_backbone = random.choice(list(BACKBONE_LIST['text'].keys())),
        video_backbone = random.choice(list(BACKBONE_LIST['video'].keys())),
        audio_backbone = random.choice(list(BACKBONE_LIST['audio'].keys())),
    )

    B, T, D = 4, 3, args.hidden_dim
    x = torch.randn(B, T, D, device=device)

    aligner = PerceptualAligner(args).to(device)
    readout = Readout(args, d_model=D, fmri_out_dim=args.num_queries).to(device)

    # Eval pass (deterministic shape check)
    aligner.eval()
    with torch.no_grad():
        aligned = aligner(x)
        fmri_pred, l2_reg = readout(aligned)

    print("=== Shape check (eval) ===")
    print(f"Input multimodal tokens: {x.shape} (expected [B, T, D])")
    print(f"Aligned tokens: {aligned.shape} (expected [B, Q, D])")
    print(f"fMRI prediction: {fmri_pred.shape} (expected [B, Q])")
    print(f"L2 reg: {l2_reg.item():.6f} (scalar tensor)")

    assert aligned.shape == (B, args.num_queries, D), "PerceptualAligner output shape mismatch"
    assert fmri_pred.shape == (B, args.num_queries), "Readout output shape mismatch"

    # Train pass (dropout active)
    aligner.train()
    aligned_train = aligner(x)
    fmri_pred_train, _ = readout(aligned_train)

    print("=== Shape check (train; modality dropout active) ===")
    print(f"Aligned tokens (train): {aligned_train.shape}")
    print(f"fMRI prediction (train): {fmri_pred_train.shape}")

    assert torch.isfinite(aligned_train).all(), "NaN/Inf detected in aligned_train"
    assert torch.isfinite(fmri_pred_train).all(), "NaN/Inf detected in fmri_pred_train"

    print("All pseudo-data shape checks passed.")

