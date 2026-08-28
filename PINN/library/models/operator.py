"""
Neural-operator style encoders plus the shared voxel MLP decoder.

References:
- Fourier Neural Operator: https://arxiv.org/abs/2010.08895
- Adaptive Fourier Neural Operator: https://arxiv.org/abs/2111.13587
- U-FNO: https://arxiv.org/abs/2109.03697
- U-shaped Neural Operators: https://arxiv.org/abs/2204.11127
- Graph Neural Operator / kernel networks: https://arxiv.org/abs/2003.03485
- Multipole Graph Neural Operator: https://arxiv.org/abs/2006.09535
- DeepONet: https://arxiv.org/abs/1910.03193

GNO/MGNO/DeepONet are not instantiated here because the current MRI pipeline is
regular-grid image-to-grid. They need graph edges or branch/trunk operator
inputs, so forcing them into this interface would hide important assumptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import ConvResidualBlock, GridEncoderMLPModel, filter_init_kwargs, make_conv_stem, make_voxel_mlp


class SpectralConv2d(nn.Module):
    """Low-mode Fourier convolution used by FNO. Paper: https://arxiv.org/abs/2010.08895"""

    def __init__(self, channels: int, modes: int = 12):
        super().__init__()
        self.modes = modes
        scale = 1.0 / max(channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        out_ft = torch.zeros(B, C, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        mh = min(self.modes, H)
        mw = min(self.modes, W // 2 + 1)
        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mh, :mw],
            self.weight[:, :, :mh, :mw],
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(x.dtype)


class SpectralConv3d(nn.Module):
    """3D low-mode Fourier convolution used by FNO. Paper: https://arxiv.org/abs/2010.08895"""

    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        self.modes = modes
        scale = 1.0 / max(channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
        out_ft = torch.zeros(B, C, D, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        md = min(self.modes, D)
        mh = min(self.modes, H)
        mw = min(self.modes, W // 2 + 1)
        out_ft[:, :, :md, :mh, :mw] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :md, :mh, :mw],
            self.weight[:, :, :md, :mh, :mw],
        )
        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1), norm="ortho").to(x.dtype)


class FNOBlock(nn.Module):
    """FNO block: spectral convolution plus pointwise residual path. Paper: https://arxiv.org/abs/2010.08895"""

    def __init__(self, channels: int, modes: int = 12, spatial_ndim: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        spectral_cls = SpectralConv2d if spatial_ndim == 2 else SpectralConv3d
        pointwise_cls = nn.Conv2d if spatial_ndim == 2 else nn.Conv3d
        self.spectral = spectral_cls(channels, modes=modes)
        self.pointwise = pointwise_cls(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()
        self.dropout = (
            nn.Dropout2d(dropout) if spatial_ndim == 2 and dropout > 0.0
            else nn.Dropout3d(dropout) if spatial_ndim == 3 and dropout > 0.0
            else nn.Identity()
        )

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.spectral(x) + self.pointwise(x))))


class FNOEncoder(nn.Module):
    """Fourier Neural Operator encoder for regular 2D/3D grids. Paper: https://arxiv.org/abs/2010.08895"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 4, modes: int = 12, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.Sequential(*[
            FNOBlock(enc_ch, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.stem(x))


class AFNOBlock(nn.Module):
    """Adaptive Fourier Neural Operator token mixer. Paper: https://arxiv.org/abs/2111.13587"""

    def __init__(self, channels: int, num_blocks: int = 8,
                 sparsity_threshold: float = 0.01, hard_thresholding_fraction: float = 1.0,
                 dropout: float = 0.0, spatial_ndim: int = 2):
        super().__init__()
        if channels % num_blocks != 0:
            raise ValueError(f"AFNO requires channels % num_blocks == 0, got {channels}, {num_blocks}.")
        self.channels = channels
        self.num_blocks = num_blocks
        self.block_size = channels // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.w1 = nn.Parameter(torch.randn(num_blocks, self.block_size, self.block_size) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_blocks, self.block_size))
        self.w2 = nn.Parameter(torch.randn(num_blocks, self.block_size, self.block_size) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_blocks, self.block_size))
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout = (
            nn.Dropout2d(dropout) if spatial_ndim == 2 and dropout > 0.0
            else nn.Dropout3d(dropout) if spatial_ndim == 3 and dropout > 0.0
            else nn.Identity()
        )

    def forward(self, x):
        dtype = x.dtype
        residual = x
        spatial = x.shape[2:]
        x_ft = torch.fft.rfftn(x.float(), dim=tuple(range(2, x.ndim)), norm="ortho")
        freq_shape = x_ft.shape[2:]
        kept = [max(1, int(size * self.hard_thresholding_fraction)) for size in freq_shape]
        slices = (slice(None), slice(None)) + tuple(slice(0, k) for k in kept)
        z = x_ft[slices].permute(0, *range(2, x_ft.ndim), 1).contiguous()
        z = z.reshape(-1, self.num_blocks, self.block_size)
        z = torch.einsum("nbi,bio->nbo", z, self.w1.to(z.dtype)) + self.b1.to(z.dtype)
        z = F.gelu(z.real).to(z.dtype) + 1j * F.gelu(z.imag).to(z.dtype)
        z = torch.einsum("nbi,bio->nbo", z, self.w2.to(z.dtype)) + self.b2.to(z.dtype)
        z = F.softshrink(z.real, lambd=self.sparsity_threshold).to(z.dtype) + \
            1j * F.softshrink(z.imag, lambd=self.sparsity_threshold).to(z.dtype)
        z = z.reshape(x_ft.shape[0], *kept, self.channels).permute(0, -1, *range(1, 1 + len(kept))).contiguous()
        out_ft = torch.zeros_like(x_ft)
        out_ft[slices] = z
        y = torch.fft.irfftn(out_ft, s=spatial, dim=tuple(range(2, x.ndim)), norm="ortho").to(dtype)
        return residual + self.dropout(self.norm(y))


class AFNOEncoder(nn.Module):
    """Regular-grid AFNO encoder with the shared convolutional stem. Paper: https://arxiv.org/abs/2111.13587"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 4, num_blocks: int = 8,
                 sparsity_threshold: float = 0.01, hard_thresholding_fraction: float = 1.0,
                 dropout: float = 0.0, stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.Sequential(*[
            AFNOBlock(
                enc_ch,
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
                dropout=dropout,
                spatial_ndim=spatial_ndim,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.stem(x))


class LocalUNetMixBlock(nn.Module):
    """Small local U-Net mixer used by U-FNO-style encoders. U-Net paper: https://arxiv.org/abs/1505.04597"""

    def __init__(self, channels: int, spatial_ndim: int = 2, use_kan: bool = False):
        super().__init__()
        pool_cls = nn.MaxPool2d if spatial_ndim == 2 else nn.MaxPool3d
        self.spatial_ndim = spatial_ndim
        self.down = ConvResidualBlock(channels, channels, spatial_ndim, use_kan=use_kan, residual=True, num_convs=2)
        self.pool = pool_cls(2)
        self.mid = ConvResidualBlock(channels, channels, spatial_ndim, use_kan=use_kan, residual=True, num_convs=2)
        self.up = ConvResidualBlock(channels * 2, channels, spatial_ndim, use_kan=use_kan, residual=True, num_convs=2)

    def forward(self, x):
        mode = "bilinear" if self.spatial_ndim == 2 else "trilinear"
        skip = self.down(x)
        y = self.mid(self.pool(skip))
        y = F.interpolate(y, size=skip.shape[2:], mode=mode, align_corners=False)
        return self.up(torch.cat([y, skip], dim=1))


class UFNOBlock(nn.Module):
    """U-FNO block: FNO global spectral path plus local U-Net path. Paper: https://arxiv.org/abs/2109.03697"""

    def __init__(self, channels: int, modes: int = 12, spatial_ndim: int = 2,
                 dropout: float = 0.0, use_kan: bool = False):
        super().__init__()
        self.fno = FNOBlock(channels, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout)
        self.local = LocalUNetMixBlock(channels, spatial_ndim=spatial_ndim, use_kan=use_kan)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.fno(x) + self.local(x)))


class UFNOEncoder(nn.Module):
    """U-FNO encoder for regular grids, combining spectral and local U-Net mixing. Paper: https://arxiv.org/abs/2109.03697"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 4, modes: int = 12, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.Sequential(*[
            UFNOBlock(enc_ch, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout, use_kan=use_kan)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.stem(x))


class UNOEncoder(nn.Module):
    """U-shaped Neural Operator style encoder with multiscale FNO blocks. Paper: https://arxiv.org/abs/2204.11127"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, modes: int = 12, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        self.out_ch = enc_ch
        self.spatial_ndim = spatial_ndim
        pool_cls = nn.AvgPool2d if spatial_ndim == 2 else nn.AvgPool3d
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.down_blocks = nn.ModuleList([
            FNOBlock(enc_ch, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout)
            for _ in range(depth)
        ])
        self.pools = nn.ModuleList([pool_cls(2) for _ in range(depth)])
        self.mid = FNOBlock(enc_ch, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout)
        self.up_blocks = nn.ModuleList([
            FNOBlock(enc_ch, modes=modes, spatial_ndim=spatial_ndim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        mode = "bilinear" if self.spatial_ndim == 2 else "trilinear"
        feat = self.stem(x)
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            feat = block(feat)
            skips.append(feat)
            feat = pool(feat)
        feat = self.mid(feat)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            feat = F.interpolate(feat, size=skip.shape[2:], mode=mode, align_corners=False)
            feat = block(feat + skip)
        return feat


class OperatorModel(GridEncoderMLPModel):
    """Neural-operator encoder plus shared MLP decoder. FNO reference: https://arxiv.org/abs/2010.08895"""

    ENCODERS = {
        "FNO": FNOEncoder,
        "AFNO": AFNOEncoder,
        "UFNO": UFNOEncoder,
        "U-FNO": UFNOEncoder,
        "UNO": UNOEncoder,
    }

    def __init__(self, n_grad: int, spatial_dim: int = 3, out_params: int = 7,
                 encoder: dict | None = None, decoder: dict | None = None,
                 use_KAN: bool = False, use_VAE: bool = False,
                 latent_dim: int = 64, vae_hidden: int = 128):
        encoder = encoder or {"type": "FNO"}
        decoder = decoder or {}
        enc_type = encoder.get("type", "FNO")
        if enc_type not in self.ENCODERS:
            raise ValueError(f"Unknown operator encoder type {enc_type!r}. Options: {tuple(self.ENCODERS)}")
        enc_cls = self.ENCODERS[enc_type]
        enc_kwargs = filter_init_kwargs(enc_cls, {k: v for k, v in encoder.items() if k != "type"})
        enc = enc_cls(n_grad=n_grad, spatial_ndim=spatial_dim, use_kan=use_KAN, **enc_kwargs)
        mlp = make_voxel_mlp(
            enc.out_ch,
            decoder.get("hidden", 256),
            decoder.get("num_layers", 5),
            out_params,
            decoder.get("neck", 64),
            decoder.get("dropout", 0.0),
            use_kan=use_KAN,
        )
        super().__init__(enc, mlp, enc.out_ch, spatial_dim,
                         use_VAE=use_VAE, latent_dim=latent_dim, vae_hidden=vae_hidden)


class FNO(OperatorModel):
    """Fourier Neural Operator model: FNO encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2010.08895"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "FNO", **(encoder or {})}, **kwargs)


class AFNO(OperatorModel):
    """Adaptive Fourier Neural Operator model: AFNO encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2111.13587"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "AFNO", **(encoder or {})}, **kwargs)


class UFNO(OperatorModel):
    """U-FNO model: spectral FNO blocks fused with local U-Net mixing. Paper: https://arxiv.org/abs/2109.03697"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "UFNO", **(encoder or {})}, **kwargs)


class UNO(OperatorModel):
    """U-shaped Neural Operator model with multiscale FNO mixing. Paper: https://arxiv.org/abs/2204.11127"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "UNO", **(encoder or {})}, **kwargs)
