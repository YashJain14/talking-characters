"""
loconet/__init__.py
--------------------
Standalone PyTorch implementation of LoCoNet (CVPR 2024) whose architecture
is reverse-engineered from the official AVA-pretrained checkpoint keys:

  model.module.model.visualFrontend.*   3D-stem + ResNet-18 (TalkNet variant)
  model.module.model.visualTCN.*        5-block dilated TCN
  model.module.model.visualConv1D.*     1D projection  512→128
  model.module.model.audioEncoder.*     VGGish-lite + deconv + conv1d
  model.module.model.crossA2V / crossV2A   TransformerEncoderLayer  (LIM)
  model.module.model.convAV.[0,2,4]    SIM conv2d blocks
  model.module.model.convAV.[1,3,5]    LIM TransformerEncoderLayer blocks
  model.module.lossAV.FC               Linear(256,2) classifier

Public API (matches active_speaker.py):

    from loconet import LoCoNet
    model = LoCoNet()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)          # strips 'model.module.' prefix
    model.eval().to(device)
    logits = model(face_tensor, mel_tensor)   # [N] logits; sigmoid → prob

    face_tensor : [N, 3, 112, 112]  float32  values in [0,1]
    mel_tensor  : [N, 9, 13]        float32  MFCC slices per frame

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
    """Per-channel LayerNorm with gamma/beta params (matches ckpt key names)."""
    def __init__(self, d: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta  = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,d,T]
        mu  = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True, unbiased=False).clamp(min=1e-5)
        return self.gamma.view(1,-1,1)*(x-mu)/std + self.beta.view(1,-1,1)


# ═══════════════════════════════════════════════════════════════
# 2.  Visual frontend
#       frontend3D : Conv3d(1,64,5×7×7) + BN3d + ReLU + MaxPool3d
#       resnet     : 4-layer ResNet-18 with two blocks (a/b) per layer
# ═══════════════════════════════════════════════════════════════

class _TwoBlockLayer(nn.Module):
    """
    Each ResNet layer holds two independent basic blocks named *a and *b.
    Layer 2-4 have a single downsample conv (key: layer.downsample.weight).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        need_ds = (stride != 1 or in_ch != out_ch)
        # block a
        self.conv1a = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1a   = nn.BatchNorm2d(out_ch)
        self.conv2a = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.outbna = nn.BatchNorm2d(out_ch)
        if need_ds:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        # block b  (same channels, no downsample)
        self.conv1b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1b   = nn.BatchNorm2d(out_ch)
        self.conv2b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # block a
        r = self.downsample(x) if hasattr(self, 'downsample') else x
        h = F.relu(self.bn1a(self.conv1a(x)), inplace=True)
        h = self.conv2a(h)
        x = F.relu(self.outbna(h + r), inplace=True)
        # block b
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,64,H,W]→[B,512]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)


class _VisualFrontend(nn.Module):
    """
    Input  : [B, 3, 112, 112]   (RGB float in [0,1])
    Output : [B, 512]
    """
    def __init__(self):
        super().__init__()
        # checkpoint indices: frontend3D.0=Conv3d  frontend3D.1=BN3d
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5,7,7),
                      stride=(1,2,2), padding=(2,3,3), bias=False),  # [0]
            nn.BatchNorm3d(64),                                        # [1]
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        )
        self.resnet = _ResNet18ASD()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB → grayscale, add temporal dim of 1
        gray = (0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2])  # [B,112,112]
        gray = gray[:, None, None]                                  # [B,1,1,112,112]
        v = self.frontend3D(gray)   # [B,64,1,H',W']
        v = v.squeeze(2)            # [B,64,H',W']
        return self.resnet(v)       # [B,512]


# ═══════════════════════════════════════════════════════════════
# 3.  Visual TCN  (5 dilated blocks)
#     per block i in checkpoint:
#       visualTCN.net.i.net.1  BN1d(512)
#       visualTCN.net.i.net.2  Conv1d(512,512,3, dilation=2^i)
#       visualTCN.net.i.net.3  Conv1d(512,512,1)
#       visualTCN.net.i.net.4  _LN1d(512)   [gamma/beta]
#       visualTCN.net.i.net.5  Conv1d(512,512,1)   [residual branch]
# ═══════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Identity(),                                                       # 0 placeholder
            nn.BatchNorm1d(d),                                                   # 1
            nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation, bias=True), # 2
            nn.Conv1d(d, d, 1, bias=True),                                       # 3
            _LN1d(d),                                                            # 4
            nn.Conv1d(d, d, 1, bias=True),                                       # 5 residual
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.net[5](x)                          # residual branch
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
# 4.  Visual Conv1D projection  512→128
#     visualConv1D.net.0  Conv1d(512,128,1)
#     visualConv1D.net.1  BN1d(128)
#     visualConv1D.net.3  Conv1d(128,128,1)
# ═══════════════════════════════════════════════════════════════

class _VisualConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Conv1d(512, 128, 1, bias=True),  # 0
            nn.BatchNorm1d(128),                 # 1
            nn.ReLU(inplace=True),               # 2  (no weights in ckpt)
            nn.Conv1d(128, 128, 1, bias=True),  # 3
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 5.  Audio encoder
#     features.{0,3,6,8,11,13}  Conv2d layers (VGGish-lite)
#     deconv                     ConvTranspose2d(512,128,4×1)
#     conv1                      Conv1d(128,128,1)
#     conv2                      Conv1d(128,128,1)
# ═══════════════════════════════════════════════════════════════

class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),     # 0
            nn.ReLU(inplace=True),               # 1
            nn.MaxPool2d(2, 2),                  # 2
            nn.Conv2d(64, 128, 3, padding=1),   # 3
            nn.ReLU(inplace=True),               # 4
            nn.MaxPool2d(2, 2),                  # 5
            nn.Conv2d(128, 256, 3, padding=1),  # 6
            nn.ReLU(inplace=True),               # 7
            nn.Conv2d(256, 256, 3, padding=1),  # 8
            nn.ReLU(inplace=True),               # 9
            nn.MaxPool2d(2, 2),                  # 10
            nn.Conv2d(256, 512, 3, padding=1),  # 11
            nn.ReLU(inplace=True),               # 12
            nn.Conv2d(512, 512, 3, padding=1),  # 13
            nn.ReLU(inplace=True),               # 14
        )
        self.deconv = nn.ConvTranspose2d(512, 128, kernel_size=(4,1), stride=(4,1), bias=True)
        self.conv1  = nn.Conv1d(128, 128, 1, bias=True)
        self.conv2  = nn.Conv1d(128, 128, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 9, 13]
        B, T, H, W = x.shape
        x = x.reshape(B*T, 1, H, W)
        x = self.features(x)                           # [B*T, 512, h', w']
        x = F.adaptive_avg_pool2d(x, (4, 1))           # [B*T, 512, 4, 1]
        x = F.relu(self.deconv(x), inplace=True)       # [B*T, 128, 16, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))           # [B*T, 128, 1, 1]
        x = x.view(B, T, 128).permute(0, 2, 1)         # [B, 128, T]
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        return x                                        # [B, 128, T]


# ═══════════════════════════════════════════════════════════════
# 6.  SIM block  (convAV even indices 0,2,4)
#     conv2d         Conv2d(256,256,(1,3), padding=(0,1))
#     ln             LayerNorm(256)
#     conv2d_1x1     Conv2d(256,256,1)
#     conv2d_1x1_2   Conv2d(256,256,1)   [residual]
# ═══════════════════════════════════════════════════════════════

class _SIMBlock(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.conv2d       = nn.Conv2d(d, d, (1,3), padding=(0,1), bias=True)
        self.ln           = nn.LayerNorm(d)
        self.conv2d_1x1   = nn.Conv2d(d, d, 1, bias=True)
        self.conv2d_1x1_2 = nn.Conv2d(d, d, 1, bias=True)   # residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d, S, T]   S = number of speakers
        r = self.conv2d_1x1_2(x)
        h = self.conv2d(x)                      # inter-speaker conv
        h = h.permute(0, 2, 3, 1)              # [B, S, T, d]
        h = self.ln(h)
        h = h.permute(0, 3, 1, 2)              # [B, d, S, T]
        return F.relu(self.conv2d_1x1(h) + r, inplace=True)


# ═══════════════════════════════════════════════════════════════
# 7.  LIM block  (convAV odd indices 1,3,5)
#     Standard TransformerEncoderLayer stored with default PyTorch names.
#     d=256, nhead=8  (in_proj_weight shape [768,256] → 3*256=768)
# ═══════════════════════════════════════════════════════════════

class _LIMBlock(nn.TransformerEncoderLayer):
    def __init__(self, d: int, nhead: int = 8, dim_ff: int = 512):
        super().__init__(d_model=d, nhead=nhead, dim_feedforward=dim_ff,
                         dropout=0.0, batch_first=False)


# ═══════════════════════════════════════════════════════════════
# 8.  Core network  (model.module.model.*)
# ═══════════════════════════════════════════════════════════════

class _LoCoNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualFrontend = _VisualFrontend()
        self.visualTCN      = _VisualTCN(d=512, n=5)
        self.visualConv1D   = _VisualConv1D()
        self.audioEncoder   = _AudioEncoder()
        # crossA2V/V2A: d=128  (in_proj_weight [384,128] → nhead must divide 128)
        self.crossA2V = _LIMBlock(d=128, nhead=8, dim_ff=512)
        self.crossV2A = _LIMBlock(d=128, nhead=8, dim_ff=512)
        # convAV: d=256  (in_proj_weight [768,256])
        self.convAV = nn.ModuleList([
            _SIMBlock(d=256),                       # 0
            _LIMBlock(d=256, nhead=8, dim_ff=512),  # 1
            _SIMBlock(d=256),                       # 2
            _LIMBlock(d=256, nhead=8, dim_ff=512),  # 3
            _SIMBlock(d=256),                       # 4
            _LIMBlock(d=256, nhead=8, dim_ff=512),  # 5
        ])


# ═══════════════════════════════════════════════════════════════
# 9.  Module wrapper  (model.module.*)
# ═══════════════════════════════════════════════════════════════

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
# 10.  Public LoCoNet class
# ═══════════════════════════════════════════════════════════════

class LoCoNet(nn.Module):
    """
    Standalone LoCoNet inference wrapper.

    Usage:
        model = LoCoNet()
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval().to(device)

        with torch.inference_mode():
            logits = model(face, mel)          # [N] logits
            probs  = torch.sigmoid(logits)     # [N] speaking probability
    """

    def __init__(self):
        super().__init__()
        self._m = _LoCoNetModule()

    # ── weight loading ─────────────────────────────────────────
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

    # ── forward ───────────────────────────────────────────────
    def forward(self, face: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """
        face : [N, 3, 112, 112]  float32  [0,1]
        mel  : [N, 9, 13]        float32
        Returns [N] logits.  Apply torch.sigmoid() to get speaking probability.
        """
        N = face.shape[0]
        if N == 0:
            return face.new_zeros(0)

        core = self._m.model

        # ── visual stream ────────────────────────────────────
        vis = core.visualFrontend(face)                # [N, 512]
        vis = vis.T.unsqueeze(0)                       # [1, 512, N]
        vis = core.visualTCN(vis)                      # [1, 512, N]
        vis = core.visualConv1D(vis)                   # [1, 128, N]

        # ── audio stream ─────────────────────────────────────
        aud = mel.unsqueeze(0)                         # [1, N, 9, 13]
        aud = core.audioEncoder(aud)                   # [1, 128, N]

        # ── LIM cross-attention (A2V, V2A) ───────────────────
        # TransformerEncoderLayer expects [T, B, d]
        v = vis.permute(2, 0, 1)                       # [N, 1, 128]
        a = aud.permute(2, 0, 1)                       # [N, 1, 128]
        v = core.crossA2V(v)
        a = core.crossV2A(a)
        vis = v.permute(1, 2, 0)                       # [1, 128, N]
        aud = a.permute(1, 2, 0)                       # [1, 128, N]

        # ── concatenate A+V ───────────────────────────────────
        av = torch.cat([vis, aud], dim=1)              # [1, 256, N]
        av = av.unsqueeze(2)                           # [1, 256, 1, N]  (S=1)

        # ── SIM/LIM interleaved stack ─────────────────────────
        for i, blk in enumerate(core.convAV):
            if i % 2 == 0:                             # SIM
                av = blk(av)                           # [1, 256, 1, N]
            else:                                      # LIM
                B, d, S, T = av.shape
                x = av.view(B*S, d, T).permute(2, 0, 1)  # [T, B*S, d]
                x = blk(x)
                av = x.permute(1, 2, 0).view(B, d, S, T)

        av = av.squeeze(2).squeeze(0)                  # [256, N]
        av = av.T                                      # [N, 256]

        # ── classifier ────────────────────────────────────────
        logits = self._m.lossAV.FC(av)                # [N, 2]
        return logits[:, 1]                            # [N]  speaking logit