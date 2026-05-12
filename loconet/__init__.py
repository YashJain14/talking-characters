"""
loconet/__init__.py
--------------------
Standalone PyTorch implementation of LoCoNet (CVPR 2024).
Architecture derived from the official AVA-pretrained checkpoint.

Public API:

    from loconet import LoCoNet
    model = LoCoNet()
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)

    # face  : [N, 1, 112, 112]  float32  grayscale, normalised (x/255-0.4161)/0.1688
    # audio : [1, T_audio, n_mels]  float32  mel spectrogram, T_audio = 4*N
    logits = model(face, audio)   # [N] logits; sigmoid → speaking prob

Zero dependencies beyond PyTorch.
"""

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1.  Helpers
# ═══════════════════════════════════════════════════════════════

class _LN1d(nn.Module):
    """Per-channel LayerNorm stored as [1, d, 1] to match checkpoint."""
    def __init__(self, d: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, d, 1))
        self.beta  = nn.Parameter(torch.zeros(1, d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu  = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True, unbiased=False).clamp(min=1e-5)
        return self.gamma * (x - mu) / std + self.beta


# ═══════════════════════════════════════════════════════════════
# 2.  Visual frontend
#     frontend3D : Conv3d(1,64,5×7×7) + BN3d + ReLU + MaxPool3d
#     resnet     : ResNet-18 style, 4 two-block layers
# ═══════════════════════════════════════════════════════════════

class _TwoBlockLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        need_ds = (stride != 1 or in_ch != out_ch)
        self.conv1a = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1a   = nn.BatchNorm2d(out_ch)
        self.conv2a = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.outbna = nn.BatchNorm2d(out_ch)
        if need_ds:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.conv1b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1b   = nn.BatchNorm2d(out_ch)
        self.conv2b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.downsample(x) if hasattr(self, 'downsample') else x
        h = F.relu(self.bn1a(self.conv1a(x)), inplace=True)
        h = self.conv2a(h)
        x = F.relu(self.outbna(h + r), inplace=True)
        r = x
        h = F.relu(self.bn1b(self.conv1b(x)), inplace=True)
        h = self.conv2b(h)
        x = F.relu(self.outbnb(h + r), inplace=True)
        return x


class _ResNet18ASD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1  = _TwoBlockLayer(64,  64,  stride=1)
        self.layer2  = _TwoBlockLayer(64,  128, stride=2)
        self.layer3  = _TwoBlockLayer(128, 256, stride=2)
        self.layer4  = _TwoBlockLayer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)


class _VisualFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5,7,7),
                      stride=(1,2,2), padding=(2,3,3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        )
        self.resnet = _ResNet18ASD()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]  C=1 (grayscale) or C=3 (RGB), normalised
        N, C, H, W = x.shape
        if C == 3:
            gray = (0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2])  # [N,H,W]
        else:
            gray = x[:,0]                    # [N, H, W]
        # Normalise: (px/255 - 0.4161)/0.1688 — applied if input is in [0,1]
        # (caller may have already normalised; we just reshape here)
        gray = gray.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, H, W]
        x = self.frontend3D(gray)              # [N, 64, 1, h, w]
        x = x.squeeze(2)                       # [N, 64, h, w]
        x = self.resnet(x)                     # [N, 512]
        return x


# ═══════════════════════════════════════════════════════════════
# 3.  Visual TCN  (5 dilated depthwise blocks)
# ═══════════════════════════════════════════════════════════════

class _ScalarScale(nn.Module):
    """Single learnable scalar — checkpoint stores net.i.net.3 as shape [1]."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _TCNBlock(nn.Module):
    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Identity(),                                                               # 0
            nn.BatchNorm1d(d),                                                           # 1
            nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation, groups=d, bias=True), # 2
            _ScalarScale(),                                                              # 3
            _LN1d(d),                                                                    # 4
            nn.Conv1d(d, d, 1, bias=True),                                               # 5 residual
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.net[5](x)
        h = self.net[1](x)
        h = F.relu(self.net[2](h), inplace=True)
        h = self.net[3](h)
        h = self.net[4](h)
        return F.relu(h + r, inplace=True)


class _VisualTCN(nn.Module):
    def __init__(self, d: int = 512, n: int = 5):
        super().__init__()
        self.net = nn.ModuleList([_TCNBlock(d, 2**i) for i in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.net:
            x = blk(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 4.  Visual Conv1D  512→256→128
# ═══════════════════════════════════════════════════════════════

class _VisualConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Conv1d(512, 256, 5, padding=2, bias=True),  # 0
            nn.BatchNorm1d(256),                             # 1
            nn.ReLU(inplace=True),                           # 2
            nn.Conv1d(256, 128, 1, bias=True),              # 3
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 5.  Audio encoder  (VGGish-lite)
#     Checkpoint conv indices: 0,3,6,8,11,13 (no BN)
#     deconv : ConvTranspose2d(512,256,2×2) — present in ckpt, unused in inference
#     conv1  : Conv2d(512,256,1)
#     conv2  : Conv2d(256,128,1)
#
#     Input:  [B, T_audio, n_mels]  where T_audio ≈ 4*T_video
#     Output: [B, T_video, 128]
# ═══════════════════════════════════════════════════════════════

class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,   64,  3, padding=1),   # 0
            nn.ReLU(inplace=True),                # 1
            nn.MaxPool2d(2, 2),                   # 2
            nn.Conv2d(64,  128, 3, padding=1),   # 3
            nn.ReLU(inplace=True),                # 4
            nn.MaxPool2d(2, 2),                   # 5
            nn.Conv2d(128, 256, 3, padding=1),   # 6
            nn.ReLU(inplace=True),                # 7
            nn.Conv2d(256, 256, 3, padding=1),   # 8
            nn.ReLU(inplace=True),                # 9
            nn.MaxPool2d(2, 2),                   # 10
            nn.Conv2d(256, 512, 3, padding=1),   # 11
            nn.ReLU(inplace=True),                # 12
            nn.Conv2d(512, 512, 3, padding=1),   # 13
            nn.ReLU(inplace=True),                # 14
        )
        self.deconv = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(2,2), bias=True)
        self.conv1  = nn.Conv2d(512, 256, 1, bias=True)
        self.conv2  = nn.Conv2d(256, 128, 1, bias=True)
        self.pool   = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_audio, n_mels]
        B, T_audio, n_mels = x.shape
        # Pad T_audio to multiple of 8 (matches original training)
        pad = (8 - T_audio % 8) % 8
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        T_padded = x.shape[1]
        n_video_frames = T_audio // 4

        # Treat [T_padded, n_mels] as a 2D spectrogram image
        x = x.unsqueeze(1)                          # [B, 1, T_padded, n_mels]
        x = self.features(x)                         # [B, 512, T', F']
        # Pool frequency dim, keep time
        B2, C, Tp, Fp = x.shape
        x = x.view(B2, C * Tp, Fp)                  # [B, 512*T', F']
        x = self.pool(x).squeeze(-1)                 # [B, 512*T']
        x = x.view(B2, 512, Tp)                      # [B, 512, T']
        # T' after 3 maxpool-by-2 on T_padded: T' = T_padded // 8
        # n_video_frames = T_audio // 4 ≈ T' * 2

        # Project 512 → 128 via 1×1 convs
        x = x.unsqueeze(-1)                          # [B, 512, T', 1]
        x = F.relu(self.conv1(x), inplace=True)      # [B, 256, T', 1]
        x = self.conv2(x).squeeze(-1)                # [B, 128, T']

        # Interpolate to match video frame count
        x = F.interpolate(x, size=n_video_frames,
                          mode='linear', align_corners=False)  # [B, 128, T_video]
        return x  # [B, 128, T_video]


# ═══════════════════════════════════════════════════════════════
# 6.  SIM block  (short-term inter-speaker convolution)
#     conv2d       : Conv2d(256, 768, (3,7))   768 = 3×256 gated
#     ln           : LayerNorm(256)
#     conv2d_1x1   : Conv2d(256, 512, 1)
#     conv2d_1x1_2 : Conv2d(512, 256, 1)
# ═══════════════════════════════════════════════════════════════

class _SIMBlock(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.conv2d       = nn.Conv2d(d, d*3, (3,7), padding=(1,3), bias=True)
        self.ln           = nn.LayerNorm(d)
        self.conv2d_1x1   = nn.Conv2d(d, d*2, 1, bias=True)
        self.conv2d_1x1_2 = nn.Conv2d(d*2, d, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d, S, T]
        h = self.conv2d(x)                           # [B, 3d, S, T]
        h1, h2, h3 = h.chunk(3, dim=1)
        h = h1 * torch.sigmoid(h2) + h3             # gated
        h = h.permute(0, 2, 3, 1)                   # [B, S, T, d]
        h = self.ln(h)
        h = h.permute(0, 3, 1, 2)                   # [B, d, S, T]
        h = F.relu(self.conv2d_1x1(h), inplace=True)
        h = self.conv2d_1x1_2(h)
        return F.relu(h + x, inplace=True)


# ═══════════════════════════════════════════════════════════════
# 7.  LIM block  (long-term intra-speaker self-attention)
# ═══════════════════════════════════════════════════════════════

class _LIMBlock(nn.TransformerEncoderLayer):
    def __init__(self, d: int, nhead: int = 8, dim_ff: int = 1024):
        super().__init__(d_model=d, nhead=nhead, dim_feedforward=dim_ff,
                         dropout=0.0, batch_first=False)


# ═══════════════════════════════════════════════════════════════
# 8.  Core network
# ═══════════════════════════════════════════════════════════════

class _LoCoNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualFrontend = _VisualFrontend()
        self.visualTCN      = _VisualTCN(d=512, n=5)
        self.visualConv1D   = _VisualConv1D()
        self.audioEncoder   = _AudioEncoder()
        self.crossA2V = _LIMBlock(d=128, nhead=8, dim_ff=512)
        self.crossV2A = _LIMBlock(d=128, nhead=8, dim_ff=512)
        self.convAV = nn.ModuleList([
            _SIMBlock(d=256),
            _LIMBlock(d=256, nhead=8, dim_ff=1024),
            _SIMBlock(d=256),
            _LIMBlock(d=256, nhead=8, dim_ff=1024),
            _SIMBlock(d=256),
            _LIMBlock(d=256, nhead=8, dim_ff=1024),
        ])


class _FCHead(nn.Module):
    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        self.FC = nn.Linear(in_d, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.FC(x)


class _LoCoNetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model  = _LoCoNetCore()
        self.lossAV = _FCHead(256, 2)
        self.lossA  = _FCHead(128, 2)
        self.lossV  = _FCHead(128, 2)


# ═══════════════════════════════════════════════════════════════
# 9.  Public LoCoNet class
# ═══════════════════════════════════════════════════════════════

class LoCoNet(nn.Module):
    """
    Standalone LoCoNet inference wrapper.

    face  : [N, 1, 112, 112]  float32  grayscale, normalised (px/255-0.4161)/0.1688
    audio : [1, T_audio, n_mels]  float32  mel spectrogram, T_audio = 4*N

    Returns [N] logits; apply torch.sigmoid() to get speaking probability.
    """

    def __init__(self):
        super().__init__()
        self._m = _LoCoNetModule()

    def load_state_dict(self, state_dict, strict: bool = True):
        prefix = "model.module."
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self._m.load_state_dict(stripped, strict=False)
        n_ok = len(stripped) - len(missing)
        if missing:
            warnings.warn(
                f"LoCoNet: {len(missing)} missing keys, {n_ok} loaded. "
                f"First 5 missing: {missing[:5]}"
            )
        if unexpected:
            warnings.warn(
                f"LoCoNet: {len(unexpected)} unexpected keys (ignored). "
                f"First 5: {unexpected[:5]}"
            )

    def forward(self, face: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        face  : [N, 1, 112, 112]  grayscale normalised
        audio : [1, T_audio, n_mels]  mel spectrogram  T_audio = 4*N
        Returns [N] speaking logits.
        """
        N = face.shape[0]
        if N == 0:
            return face.new_zeros(0)

        core = self._m.model

        # ── visual stream ──────────────────────────────────────
        vis = core.visualFrontend(face)                 # [N, 512]
        vis = vis.T.unsqueeze(0)                        # [1, 512, N]
        vis = core.visualTCN(vis)                       # [1, 512, N]
        vis = core.visualConv1D(vis)                    # [1, 128, N]

        # ── audio stream ───────────────────────────────────────
        aud = core.audioEncoder(audio)                  # [1, 128, N]

        # ── cross-attention ────────────────────────────────────
        # [N, 1, 128] — seq_len=N, batch=1, dim=128
        v = vis.permute(2, 0, 1)
        a = aud.permute(2, 0, 1)
        v = core.crossA2V(v)
        a = core.crossV2A(a)
        vis = v.permute(1, 2, 0)                        # [1, 128, N]
        aud = a.permute(1, 2, 0)                        # [1, 128, N]

        # ── SIM/LIM stack ──────────────────────────────────────
        av = torch.cat([vis, aud], dim=1)               # [1, 256, N]
        av = av.unsqueeze(2)                            # [1, 256, 1, N]  S=1

        for i, blk in enumerate(core.convAV):
            if i % 2 == 0:                              # SIM block
                av = blk(av)
            else:                                       # LIM block
                B, d, S, T = av.shape
                x = av.view(B * S, d, T).permute(2, 0, 1)  # [T, B*S, d]
                x = blk(x)
                av = x.permute(1, 2, 0).view(B, d, S, T)

        av = av.squeeze(2).squeeze(0).T                 # [N, 256]
        logits = self._m.lossAV.FC(av)                  # [N, 2]
        return logits[:, 1]                              # [N] speaking logit
