"""
Composable implicit neural representation (INR) models for 3D coordinates.

The main entry point is CoordINR:

    coords -> CoordEncoder -> CoordDecoder -> params

coords are supplied by data.dataset=3Dcoord as normalized voxel coordinates.
The network predicts raw physical-parameter outputs; the physics module later
maps those outputs into DTI/DKI parameters and then reconstructs DWI signals.

Config structure:

    model:
      name: coords
      coords:
        input:  {}
        encoder:{mode: hash_grid, ...}
        decoder:{activation: relu, residual: false, ...}

Hash, Fourier, SIREN, grid, and tri-plane INR variants are configured through
model.coords.encoder / model.coords.decoder rather than separate model classes.
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

from .components import KANMLP, PointEncoderMLPModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "softplus":
        return nn.Softplus()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class Sine(nn.Module):
    """Periodic activation used by SIREN. Paper: https://arxiv.org/abs/2006.09661"""

    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


def _make_activation(name: str, omega: float = 30.0):
    name = name.lower()
    return Sine(omega) if name == "sine" else _activation(name)


def _filter_kwargs(cls, cfg):
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in cfg.items() if k in allowed}


# ---------------------------------------------------------------------------
# 1. CoordEncoder
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Deterministic sin/cos positional encoding. Transformer reference: https://arxiv.org/abs/1706.03762"""

    def __init__(self, num_frequencies: int = 8,
                 max_frequency: float = 64.0, include_input: bool = True):
        super().__init__()
        self.in_dim = 3
        self.num_frequencies = int(num_frequencies)
        self.include_input = include_input
        if self.num_frequencies > 0:
            bands = torch.logspace(
                0.0,
                math.log2(float(max_frequency)),
                steps=self.num_frequencies,
                base=2.0,
            )
        else:
            bands = torch.empty(0)
        self.register_buffer("frequency_bands", bands, persistent=False)
        self.out_dim = (self.in_dim if include_input else 0) + 2 * self.in_dim * self.num_frequencies

    def forward(self, x):
        x = x.float()
        parts = [x] if self.include_input else []
        if self.num_frequencies > 0:
            angles = x.unsqueeze(-2) * self.frequency_bands.view(1, -1, 1) * math.pi
            parts.extend([torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)])
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


class GaussianFourierFeatures(nn.Module):
    """Random Fourier feature encoder. Paper: https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html"""

    def __init__(self, mapping_size: int = 128,
                 scale: float = 10.0, include_input: bool = True):
        super().__init__()
        self.in_dim = 3
        self.include_input = include_input
        B = torch.randn(self.in_dim, mapping_size) * float(scale)
        self.register_buffer("B", B, persistent=True)
        self.out_dim = (self.in_dim if include_input else 0) + 2 * mapping_size

    def forward(self, x):
        proj = 2.0 * math.pi * x.float() @ self.B
        enc = [torch.sin(proj), torch.cos(proj)]
        if self.include_input:
            enc.insert(0, x.float())
        return torch.cat(enc, dim=-1)


class SirenFirstLayer(nn.Module):
    """SIREN-style first layer: sin(omega0 * Linear(x)). Paper: https://arxiv.org/abs/2006.09661"""

    def __init__(self, out_dim: int = 256, omega0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(3, out_dim)
        self.omega0 = omega0
        bound = 1.0 / 3
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)
        self.out_dim = out_dim

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x.float()))


class PlaneConcatEncoder(nn.Module):
    """Explicitly concatenate coordinate planes; related to tri-plane features: https://arxiv.org/abs/2112.07945"""

    def __init__(self, include_xyz: bool = False):
        super().__init__()
        self.include_xyz = include_xyz
        self.out_dim = 9 if include_xyz else 6

    def forward(self, coords):
        x, y, z = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
        planes = torch.cat([x, y, y, z, x, z], dim=-1)
        if self.include_xyz:
            return torch.cat([coords.float(), planes], dim=-1)
        return planes


class DenseGridEncoder(nn.Module):
    """Dense learnable 3D feature grid sampled at xyz coordinates; grid sampling reference: https://arxiv.org/abs/1506.02025"""

    def __init__(self, resolution: int = 32,
                 features: int = 8, interpolation: str = "bilinear"):
        super().__init__()
        self.interpolation = interpolation
        self.grid = nn.Parameter(torch.zeros(1, features, resolution, resolution, resolution))
        nn.init.normal_(self.grid, std=0.01)
        self.out_dim = features

    def forward(self, coords):
        sample = coords.float().clamp(0.0, 1.0) * 2.0 - 1.0
        sample = sample.view(1, -1, 1, 1, 3)
        out = F.grid_sample(
            self.grid,
            sample,
            mode=self.interpolation,
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1).contiguous()


class MultiResGridEncoder(nn.Module):
    """Concatenate several dense grids at different resolutions; hash-grid reference: https://arxiv.org/abs/2201.05989"""

    def __init__(self, resolutions=None, n_features_per_level: int = 4):
        super().__init__()
        if resolutions is None:
            resolutions = [16, 32, 64]
        self.levels = nn.ModuleList([
            DenseGridEncoder(resolution=int(r), features=n_features_per_level)
            for r in resolutions
        ])
        self.out_dim = len(resolutions) * n_features_per_level

    def forward(self, coords):
        return torch.cat([level(coords) for level in self.levels], dim=-1)


class HashGridEncoder(nn.Module):
    """tiny-cuda-nn multi-resolution hash-grid encoder. Paper: https://arxiv.org/abs/2201.05989"""

    def __init__(self, n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 per_level_scale: float = 1.5):
        super().__init__()
        if tcnn is None:
            raise RuntimeError(
                "HashGridEncoder requires tiny-cuda-nn with a CUDA-capable PyTorch environment. "
                f"Original import error: {_TCNN_IMPORT_ERROR}"
            )
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.out_dim = self.encoding.n_output_dims

    def forward(self, x):
        return self.encoding(x.float().clamp(0.0, 1.0)).float()


class TriPlaneEncoder(nn.Module):
    """Learnable XY/YZ/XZ plane features sampled and concatenated. EG3D paper: https://arxiv.org/abs/2112.07945"""

    def __init__(self, resolution: int = 64,
                 features_per_plane: int = 8):
        super().__init__()
        self.xy = nn.Parameter(torch.zeros(1, features_per_plane, resolution, resolution))
        self.yz = nn.Parameter(torch.zeros(1, features_per_plane, resolution, resolution))
        self.xz = nn.Parameter(torch.zeros(1, features_per_plane, resolution, resolution))
        for p in (self.xy, self.yz, self.xz):
            nn.init.normal_(p, std=0.01)
        self.out_dim = 3 * features_per_plane

    @staticmethod
    def _sample(plane, uv):
        grid = uv.float().clamp(0.0, 1.0) * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 2)
        out = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()

    def forward(self, x):
        xyz = x[..., :3]
        x0, y0, z0 = xyz[..., 0:1], xyz[..., 1:2], xyz[..., 2:3]
        return torch.cat([
            self._sample(self.xy, torch.cat([x0, y0], dim=-1)),
            self._sample(self.yz, torch.cat([y0, z0], dim=-1)),
            self._sample(self.xz, torch.cat([x0, z0], dim=-1)),
        ], dim=-1)


class ConcatEncoder(nn.Module):
    """Concatenate coordinate encoders; Fourier/SIREN/hash-grid references live on the child encoders."""

    def __init__(self, *encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.out_dim = sum(e.out_dim for e in self.encoders)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.encoders], dim=-1)


class CoordEncoder(nn.Module):
    """
    Encoder factory for coordinate features.
    INR/SIREN reference: https://arxiv.org/abs/2006.09661
    Hash-grid reference: https://arxiv.org/abs/2201.05989

    Modes:
      none, positional_encoding, gaussian_fourier, siren_first_layer,
      xy_yz_xz, xyz_plus_xy_yz_xz, dense_grid, multi_res_grid, hash_grid, tri_plane,
      or a list of encoder specs to concatenate.
    """

    def __init__(self, mode="hash_grid", **kwargs):
        super().__init__()
        self.mode = mode
        self.encoder = self._build_any(mode, kwargs)
        self.out_dim = self.encoder.out_dim

    def _build_any(self, mode, cfg):
        if isinstance(mode, (list, tuple)):
            encoders = []
            for item in mode:
                if isinstance(item, str):
                    encoders.append(self._build_single(item, {}))
                elif isinstance(item, dict):
                    item_mode = item.get("mode")
                    if item_mode is None:
                        raise ValueError(f"Encoder spec missing 'mode': {item}")
                    item_cfg = {k: v for k, v in item.items() if k != "mode"}
                    encoders.append(self._build_single(item_mode, item_cfg))
                else:
                    raise TypeError(f"Encoder mode list entries must be str or dict, got {type(item)}")
            return ConcatEncoder(*encoders)
        return self._build_single(mode, cfg)


    @staticmethod
    def _build_single(mode, cfg):
        if mode == "none":
            enc = nn.Identity()
            enc.out_dim = 3
            return enc
        if mode == "positional_encoding":
            return PositionalEncoding(**_filter_kwargs(PositionalEncoding, cfg))
        if mode == "gaussian_fourier":
            return GaussianFourierFeatures(**_filter_kwargs(GaussianFourierFeatures, cfg))
        if mode == "siren_first_layer":
            return SirenFirstLayer(**_filter_kwargs(SirenFirstLayer, cfg))
        if mode == "xy_yz_xz":
            return PlaneConcatEncoder(include_xyz=False)
        if mode == "xyz_plus_xy_yz_xz":
            return PlaneConcatEncoder(include_xyz=True)
        if mode == "dense_grid":
            return DenseGridEncoder(**_filter_kwargs(DenseGridEncoder, cfg))
        if mode == "multi_res_grid":
            return MultiResGridEncoder(**_filter_kwargs(MultiResGridEncoder, cfg))
        if mode == "hash_grid":
            return HashGridEncoder(**_filter_kwargs(HashGridEncoder, cfg))
        if mode == "tri_plane":
            return TriPlaneEncoder(**_filter_kwargs(TriPlaneEncoder, cfg))
        raise ValueError(f"Unknown coord encoder mode '{mode}'.")

    def forward(self, x):
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 2. CoordDecoder
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual MLP block. ResNet reference: https://arxiv.org/abs/1512.03385"""

    def __init__(self, hidden: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.dropout(self.activation(self.fc1(x)))
        y = self.fc2(y)
        return self.activation(x + y)


class CoordDecoder(nn.Module):
    """
    Decoder MLP from encoded features to raw parameter outputs.
    MLP/backprop reference: https://www.nature.com/articles/323533a0
    KAN paper: https://arxiv.org/abs/2404.19756

    activation and residual are independent:
      activation: relu, softplus, silu, gelu, sine, leakyrelu
      residual: true/false
    """

    def __init__(self, activation: str = "relu", residual: bool = False,
                 in_dim: int = 3,
                 out_dim: int = 7, hidden: int = 256,
                 num_layers: int = 4, dropout: float = 0.0,
                 omega: float = 30.0, output_init_std: float = 0.01,
                 use_KAN: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        activation = activation.lower()
        self.activation_name = activation
        self.residual = bool(residual)

        if use_KAN:
            self.net = KANMLP(
                in_dim,
                hidden,
                num_layers,
                out_dim,
                neck_ch=hidden,
                dropout=dropout,
            )
        elif self.residual:
            self.net = self._make_residual_mlp(
                in_dim, out_dim, hidden, num_layers, activation, omega, dropout, output_init_std
            )
        else:
            self.net = self._make_mlp(
                in_dim, out_dim, hidden, num_layers, activation, omega, dropout, output_init_std
            )

    @staticmethod
    def _make_mlp(in_dim, out_dim, hidden, num_layers, activation, omega, dropout, output_init_std):
        layers = []
        ch = in_dim
        for _ in range(num_layers):
            fc = nn.Linear(ch, hidden)
            nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            nn.init.zeros_(fc.bias)
            layers.extend([fc, _make_activation(activation, omega)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            ch = hidden
        head = nn.Linear(hidden, out_dim)
        nn.init.normal_(head.weight, std=output_init_std)
        nn.init.zeros_(head.bias)
        layers.append(head)
        return nn.Sequential(*layers)

    @staticmethod
    def _make_residual_mlp(in_dim, out_dim, hidden, num_layers, activation, omega, dropout, output_init_std):
        layers = []
        first = nn.Linear(in_dim, hidden)
        nn.init.kaiming_normal_(first.weight, nonlinearity="relu")
        nn.init.zeros_(first.bias)
        layers.extend([first, _make_activation(activation, omega)])
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden, _make_activation(activation, omega), dropout))
        head = nn.Linear(hidden, out_dim)
        nn.init.normal_(head.weight, std=output_init_std)
        nn.init.zeros_(head.bias)
        layers.append(head)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CoordINR(PointEncoderMLPModel):
    """Coordinate INR model: coordinate encoder plus decoder. INR/SIREN reference: https://arxiv.org/abs/2006.09661"""

    def __init__(self, n_grad=None, out_params: int = 7,
                 input: dict = None,
                 encoder: dict = None,
                 decoder: dict = None,
                 use_VAE: bool = False, latent_dim: int = 64, vae_hidden: int = 128,
                 **kwargs):
        input = input or {}
        encoder = encoder or {"mode": "hash_grid"}
        decoder = decoder or {"activation": "relu"}

        encoder_mode = encoder.get("mode", "hash_grid")

        encoder_kwargs = {k: v for k, v in encoder.items() if k != "mode"}

        coord_encoder = CoordEncoder(encoder_mode, **encoder_kwargs)
        coord_decoder = CoordDecoder(
            in_dim=coord_encoder.out_dim,
            out_dim=out_params,
            **decoder,
        )
        super().__init__(
            coord_encoder,
            coord_decoder,
            enc_ch=coord_encoder.out_dim,
            use_VAE=use_VAE,
            latent_dim=latent_dim,
            vae_hidden=vae_hidden,
        )
