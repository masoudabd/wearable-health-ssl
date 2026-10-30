"""FORGE encoder (Nisimov, Salomon, Gazit et al.; model Liornis/forge-fog, code github.com/Lior-Nis/forge-public at
b9215b9, tag npj-2026-final-submission-v2), rebuilt for frozen use without its training framework.

The classes below follow the FORGE source (src/model/preprocessors.py, transforms.py, cwt_gpu.py, backbones.py),
MIT License, Copyright (c) 2026 Lior Nisimov; the license text is in data/raw/forge/code/forge-public/LICENSE.
Differences from the source, none of which change the computation: no Hydra/Lightning wiring; the filters run with
scipy.signal.lfilter forward and backward (the algorithm of torchaudio.functional.filtfilt, which FORGE uses, in
float64 instead of float32); the wavelet buffers are read from the released weights instead of being rebuilt with
PyWavelets.

Encoder path of the medium-context (MC) model, as pretrained (experiment pretraining/spectral_patch_mae_medcontext_daily,
preprocessors "minimal"): 500 samples at 100 Hz (5 s), three axes V, ML, AP -> 0.5 Hz high-pass and 10 Hz low-pass
(4th-order Butterworth, zero phase) -> per-window, per-channel z-score -> Morlet wavelet power at 100 scales (0.5 to
50 Hz), log1p, per-window instance normalization -> 2D patch tokens (20 scales x 50 samples: 5 x 10 = 50 tokens) ->
4-layer transformer (512 wide, ALiBi bias) -> tokens (50 x 512). The frozen embedding is the token mean (512).

Because each window is z-scored per channel and the transformer sees wavelet power, the representation does not change
when an axis is negated or rescaled (tested), so FORGE is unaffected by the axis-sign differences between datasets.

ForgeDetector rebuilds the released DeFOG *detectors* (classification/*.safetensors), which carry their own encoder
copy and a BiGRU head. Their path differs from the frozen-encoder path above in the preprocessors
(configs/model/preprocessors/hierarchical_norm_pure.yaml: the same two filters, then a fixed training-fold z-score,
then per-window RevIN with learned affine parameters instead of the plain z-score) and in what follows the
transformer: the 50 tokens are folded back to 5 frequency by 10 time positions and averaged over frequency
(pipeline/classification.py), then a one-layer bidirectional GRU and a three-layer projection give two logits. The
authors' own evaluation (scripts/eval/eval_comprehensive.py) reads P(FOG) as softmax(logits)[1], while the training
loss (FogRatioLoss) uses logit 1 alone, so sigmoid(logit 1) is the alternative reading; both are returned.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

FS = 100.0


def _next_fast_len(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))


class GpuCwt(nn.Module):
    """FORGE's FFT-based CWT (cwt_gpu.py), with its buffers given instead of computed with PyWavelets."""

    def __init__(self, int_psi_scales, psi_lengths, sqrt_scales, scales, max_scales_per_batch: int = 16):
        super().__init__()
        self.register_buffer("int_psi_scales", int_psi_scales.float())
        self.register_buffer("psi_lengths", psi_lengths.long())
        self.register_buffer("sqrt_scales", sqrt_scales.float())
        self.register_buffer("scales", scales.float())
        self.max_scales_per_batch = max_scales_per_batch
        self._group_cache: dict[int, list] = {}

    def _build_groups(self, data_len: int) -> list:
        psi_lens = self.psi_lengths.cpu().numpy()
        fft_sizes = np.array([_next_fast_len(data_len + int(pl) - 1) for pl in psi_lens])
        trim_floor = np.floor((psi_lens - 2) / 2.0).astype(np.int64)
        groups = []
        arange_T = torch.arange(data_len).long()
        for fs in np.unique(fft_sizes):
            indices = np.where(fft_sizes == fs)[0]
            starts = torch.from_numpy(trim_floor[indices]).long()
            groups.append({"fft_size": int(fs), "indices": torch.from_numpy(indices).long(),
                           "gather_idx": starts.unsqueeze(1) + arange_T.unsqueeze(0),
                           "max_valid_len": int(data_len + psi_lens[indices].max() - 2)})
        return groups

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data [batch, time] -> [n_scales, batch, time] real CWT coefficients."""
        B, T = data.shape
        device = data.device
        if T not in self._group_cache:
            self._group_cache[T] = self._build_groups(T)
        out = torch.empty(self.scales.shape[0], B, T, device=device, dtype=data.dtype)
        msb = self.max_scales_per_batch
        for g in self._group_cache[T]:
            fft_size, idx = g["fft_size"], g["indices"].to(device)
            gather_idx, max_valid = g["gather_idx"].to(device), g["max_valid_len"]
            fft_data = torch.fft.fft(data, n=fft_size, dim=-1)
            for sb_start in range(0, idx.shape[0], msb):
                sb_end = min(sb_start + msb, idx.shape[0])
                sb_idx, sb_n = idx[sb_start:sb_end], sb_end - sb_start
                fft_wav = torch.fft.fft(self.int_psi_scales[sb_idx], n=fft_size, dim=-1)
                conv = torch.fft.ifft(fft_wav[:, None, :] * fft_data[None, :, :], dim=-1).real
                coef = torch.diff(conv[..., : max_valid + 1], dim=-1)
                coef = -self.sqrt_scales[sb_idx].view(sb_n, 1, 1) * coef
                sb_gi = gather_idx[sb_start:sb_end].unsqueeze(1).expand(sb_n, B, T)
                out[sb_idx] = torch.gather(coef, 2, sb_gi)
        return out


class SpectralPatchEncoder(nn.Module):
    """FORGE's 2D patch encoder (backbones.py), unchanged."""

    def __init__(self, freq_patch=20, time_patch=50, embed_dim=512, num_heads=8, vit_depth=4, dropout=0.1,
                 use_alibi=True, input_channels=3):
        super().__init__()
        self.freq_patch, self.time_patch, self.embed_dim = freq_patch, time_patch, embed_dim
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=(freq_patch, time_patch),
                                     stride=(freq_patch, time_patch))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=vit_depth)
        self.use_alibi = use_alibi
        self._alibi_cache: dict = {}

    def _alibi(self, n: int, device) -> torch.Tensor:
        if n not in self._alibi_cache:
            h = self.transformer.layers[0].self_attn.num_heads
            slopes = torch.pow(2, -torch.arange(1, h + 1, dtype=torch.float32) * 8 / h)
            pos = torch.arange(n, dtype=torch.float32)
            rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            self._alibi_cache[n] = -slopes.unsqueeze(-1).unsqueeze(-1) * rel.unsqueeze(0)
        return self._alibi_cache[n].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        x = x[:, :, : (H // self.freq_patch) * self.freq_patch, : (W // self.time_patch) * self.time_patch]
        tokens = self.patch_embed(x)
        B = tokens.shape[0]
        tokens = self.dropout(self.norm(tokens.flatten(2).permute(0, 2, 1)))
        N = tokens.shape[1]
        if self.use_alibi:
            h = self.transformer.layers[0].self_attn.num_heads
            mask = self._alibi(N, tokens.device).unsqueeze(0).expand(B, -1, -1, -1).reshape(B * h, N, N)
            return self.transformer(tokens, mask=mask)
        return self.transformer(tokens)


@torch.no_grad()
def _spectrogram(cwt: GpuCwt, x: torch.Tensor) -> torch.Tensor:
    """FORGE's WaveletTransform.forward: reflect padding, CWT, power, log1p, per-window instance norm."""
    B, C, L = x.shape
    pad = min(L // 4, L // 8)
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    coefs = cwt(x_pad.reshape(-1, x_pad.size(-1))).permute(1, 0, 2)[..., pad:-pad]
    lp = torch.log1p(coefs.abs().pow(2)).view(B, C, *coefs.shape[-2:])
    mu = lp.mean(dim=(2, 3), keepdim=True)
    sd = lp.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return ((lp - mu) / sd).float()


def _zero_phase(x: np.ndarray, taps) -> np.ndarray:
    """lfilter forward then backward with zero initial state, the algorithm of torchaudio.functional.filtfilt."""
    y = x.astype(np.float64)
    for b, a in taps:
        y = lfilter(b, a, y, axis=-1)
        y = lfilter(b, a, y[..., ::-1], axis=-1)[..., ::-1]
    return y


class ForgeEncoder(nn.Module):
    """Released FORGE encoder: (B, 3, 500) acceleration at 100 Hz (any units, axes V, ML, AP) -> tokens (B, 50, 512)."""

    def __init__(self, weights: dict):
        super().__init__()
        self.hp_b = weights["preprocessors.preprocessors.0.b_coeffs"].double().numpy()
        self.hp_a = weights["preprocessors.preprocessors.0.a_coeffs"].double().numpy()
        self.lp_b = weights["preprocessors.preprocessors.1.b_coeffs"].double().numpy()
        self.lp_a = weights["preprocessors.preprocessors.1.a_coeffs"].double().numpy()
        self.cwt = GpuCwt(weights["transform.gpu_cwt.int_psi_scales"], weights["transform.gpu_cwt.psi_lengths"],
                          weights["transform.gpu_cwt.sqrt_scales"], weights["transform.gpu_cwt.scales"])
        self.backbone = SpectralPatchEncoder()
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in weights.items()
                                       if k.startswith("backbone.")}, strict=True)
        self.eval()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """Zero-phase high-pass then low-pass (lfilter forward and backward, zero initial state, as
        torchaudio.functional.filtfilt), then per-window, per-channel z-score (unbiased SD, as torch.std)."""
        y = _zero_phase(x, ((self.hp_b, self.hp_a), (self.lp_b, self.lp_a)))
        y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        mean, std = y.mean(dim=2, keepdim=True), y.std(dim=2, keepdim=True).clamp_min(1e-8)
        return (y - mean) / std

    @torch.no_grad()
    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return _spectrogram(self.cwt, x)

    @torch.no_grad()
    def tokens(self, x: np.ndarray, device: str = "cpu") -> torch.Tensor:
        self.to(device)
        return self.backbone(self.spectrogram(self.preprocess(x).to(device)))


class RnnHead(nn.Module):
    """FORGE's RNNHead (heads.py) as the released detectors use it: sequence_output False, one bidirectional
    GRU layer of width input_dim, then three projections to two logits. Dropout is inert in eval mode."""

    def __init__(self, input_dim: int = 512, num_classes: int = 2):
        super().__init__()
        out_dim = input_dim * 2  # bidirectional
        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=1, bidirectional=True,
                          batch_first=True)
        self.output_layer_norm = nn.LayerNorm(out_dim)
        self.output_layers = nn.ModuleList([
            nn.Linear(out_dim, out_dim // 2), nn.LayerNorm(out_dim // 2), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(out_dim // 2, out_dim // 4), nn.LayerNorm(out_dim // 4), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(out_dim // 4, num_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.input_layer_norm(x))
        out = self.output_layer_norm(out).mean(dim=1)
        for layer in self.output_layers:
            out = layer(out)
        return out


class ForgeDetector(nn.Module):
    """A released FORGE classification checkpoint: (B, 3, L) acceleration at 100 Hz -> two logits per window.

    L is the context the checkpoint was trained with (500 for mc). Preprocessing follows
    hierarchical_norm_pure.yaml: the two Butterworth filters, the fixed training-fold z-score stored in the
    checkpoint, then RevIN (per-window, per-channel centring and scaling with the population SD and eps 1e-5,
    then the learned affine). RevIN undoes the fixed z-score, so the detector, like the frozen encoder, does not
    depend on the units of its input.
    """

    def __init__(self, weights: dict, metadata: dict | None = None):
        super().__init__()
        self.metadata = dict(metadata or {})
        p = "preprocessors.preprocessors."
        self.taps = ((weights[p + "0.b_coeffs"].double().numpy(), weights[p + "0.a_coeffs"].double().numpy()),
                     (weights[p + "1.b_coeffs"].double().numpy(), weights[p + "1.a_coeffs"].double().numpy()))
        self.register_buffer("pat_mean", weights[p + "2.normalizer.mean"].float().view(1, -1, 1))
        self.register_buffer("pat_std", weights[p + "2.normalizer.std"].float().view(1, -1, 1))
        self.register_buffer("revin_weight", weights[p + "3.normalizer.affine_weight"].float().view(1, -1, 1))
        self.register_buffer("revin_bias", weights[p + "3.normalizer.affine_bias"].float().view(1, -1, 1))
        self.cwt = GpuCwt(weights["transform.gpu_cwt.int_psi_scales"], weights["transform.gpu_cwt.psi_lengths"],
                          weights["transform.gpu_cwt.sqrt_scales"], weights["transform.gpu_cwt.scales"])
        self.backbone = SpectralPatchEncoder()
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in weights.items()
                                       if k.startswith("backbone.")}, strict=True)
        self.head = RnnHead()
        self.head.load_state_dict({k[len("head."):]: v for k, v in weights.items() if k.startswith("head.")},
                                  strict=True)
        self.eval()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        y = _zero_phase(x, self.taps)
        y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        y = (y - self.pat_mean.cpu()) / (self.pat_std.cpu() + 1e-8)       # PatientLevelNormalizer, eps 1e-8
        centred = y - y.mean(dim=2, keepdim=True)                          # RevIN, population SD and eps 1e-5
        sd = torch.sqrt(centred.pow(2).mean(dim=2, keepdim=True) + 1e-5)
        return (centred / sd) * self.revin_weight.cpu() + self.revin_bias.cpu()

    @torch.no_grad()
    def logits(self, x: np.ndarray, device: str = "cpu") -> torch.Tensor:
        self.to(device)
        spec = _spectrogram(self.cwt, self.preprocess(x).to(device))
        tokens = self.backbone(spec)
        B, N, D = tokens.shape
        nW = spec.shape[3] // self.backbone.time_patch
        tokens = tokens.view(B, N // nW, nW, D).mean(dim=1)  # freq-major tokens, averaged over frequency
        return self.head(tokens)


def load_forge_detector(name: str = "mc_probe_fold0", root: Path | None = None) -> ForgeDetector:
    from safetensors import safe_open
    from safetensors.torch import load_file

    root = root or Path(__file__).resolve().parents[3] / "data" / "raw" / "forge" / "model" / "classification"
    path = root / f"{name}.safetensors"
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    if meta.get("kind") != "classification":
        raise ValueError(f"{path}: not a released FORGE classification checkpoint")
    return ForgeDetector(load_file(str(path), device="cpu"), meta)


@torch.no_grad()
def forge_detector_probs(det: ForgeDetector, windows: np.ndarray, batch_size: int = 256,
                         device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """windows (N, L, 3) at 100 Hz -> (softmax(logits)[:, 1], sigmoid(logits[:, 1])), both (N,).

    The first is the authors' own evaluation reading (eval_comprehensive.py), the second the one their training
    loss implies (FogRatioLoss uses logit 1 alone).
    """
    soft, sig = [], []
    for i in range(0, len(windows), batch_size):
        xb = np.ascontiguousarray(windows[i:i + batch_size].transpose(0, 2, 1))
        lg = det.logits(xb, device=device).float().cpu()
        soft.append(torch.softmax(lg, dim=-1)[:, 1].numpy())
        sig.append(torch.sigmoid(lg[:, 1]).numpy())
    empty = np.zeros(0, np.float32)
    return (np.concatenate(soft) if soft else empty, np.concatenate(sig) if sig else empty)


def load_forge(context: str = "mc", root: Path | None = None) -> ForgeEncoder:
    from safetensors.torch import load_file

    root = root or Path(__file__).resolve().parents[3] / "data" / "raw" / "forge" / "model" / "encoders"
    return ForgeEncoder(load_file(str(root / f"{context}.safetensors"), device="cpu"))


@torch.no_grad()
def forge_embed(enc: ForgeEncoder, windows: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """windows (N, 500, 3) at 100 Hz -> (N, 512) mean-pooled tokens."""
    out = []
    for i in range(0, len(windows), batch_size):
        xb = np.ascontiguousarray(windows[i:i + batch_size].transpose(0, 2, 1))
        out.append(enc.tokens(xb, device=device).mean(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
