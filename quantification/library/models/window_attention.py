import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import GridEncoderMLPModel, filter_init_kwargs, make_conv_stem, make_voxel_mlp


@dataclass
class WindowMeta:
    """Shape metadata for reversible window token grouping."""

    spatial: tuple
    padded_spatial: tuple
    pad: list
    batch: int
    mask: torch.Tensor | None = None
    view_shape: tuple | None = None
    permute: list | None = None
    inverse_permute: list | None = None
    group_shape: tuple | None = None
    token_shape: tuple | None = None


class WindowIndexer(nn.Module):
    """Base token grouping interface. Window attention reference: https://arxiv.org/abs/2103.14030"""

    def partition(self, x):
        raise NotImplementedError

    def reverse(self, windows, meta: WindowMeta):
        raise NotImplementedError


class SwinWindowIndexer(WindowIndexer):
    """Regular/shifted window grouper used by Swin. Paper: https://arxiv.org/abs/2103.14030"""

    def __init__(self, window_size: int = 4, shift_size: int = 0):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

    @staticmethod
    def _window_partition(x, window_size: int):
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        nd = len(spatial)
        shape = [B]
        for size in spatial:
            shape.extend([size // window_size, window_size])
        shape.append(C)
        x = x.view(*shape)
        permute = [0] + [1 + 2 * i for i in range(nd)] + [2 + 2 * i for i in range(nd)] + [2 * nd + 1]
        return x.permute(*permute).contiguous().view(-1, window_size ** nd, C)

    @staticmethod
    def _window_reverse(windows, window_size: int, spatial, batch: int):
        nd = len(spatial)
        C = windows.shape[-1]
        grid = [size // window_size for size in spatial]
        shape = [batch] + grid + [window_size] * nd + [C]
        x = windows.view(*shape)
        permute = [0]
        for i in range(nd):
            permute.extend([1 + i, 1 + nd + i])
        permute.append(1 + 2 * nd)
        return x.permute(*permute).contiguous().view(batch, *spatial, C)

    def _pad(self, x):
        spatial = x.shape[1:-1]
        pad = [(self.window_size - (size % self.window_size)) % self.window_size for size in spatial]
        if any(pad):
            if len(spatial) == 2:
                x = F.pad(x, (0, 0, 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
        return x, pad

    def _attention_mask(self, spatial, device):
        shift = min(self.shift_size, self.window_size // 2)
        if shift <= 0:
            return None
        img_mask = torch.zeros((1, *spatial, 1), device=device)
        slices = []
        for _ in spatial:
            slices.append((
                slice(0, -self.window_size),
                slice(-self.window_size, -shift),
                slice(-shift, None),
            ))
        cnt = 0
        for index in itertools.product(*slices):
            img_mask[(slice(None), *index, slice(None))] = cnt
            cnt += 1
        mask_windows = self._window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def partition(self, x):
        spatial = tuple(x.shape[1:-1])
        x, pad = self._pad(x)
        padded_spatial = tuple(x.shape[1:-1])
        shift = min(self.shift_size, self.window_size // 2)
        if shift > 0:
            dims = tuple(range(1, 1 + len(padded_spatial)))
            x = torch.roll(x, shifts=tuple(-shift for _ in padded_spatial), dims=dims)
        windows = self._window_partition(x, self.window_size)
        meta = WindowMeta(
            spatial=spatial,
            padded_spatial=padded_spatial,
            pad=pad,
            batch=x.shape[0],
            mask=self._attention_mask(padded_spatial, x.device),
        )
        return windows, meta

    def reverse(self, windows, meta: WindowMeta):
        x = self._window_reverse(windows, self.window_size, meta.padded_spatial, meta.batch)
        shift = min(self.shift_size, self.window_size // 2)
        if shift > 0:
            dims = tuple(range(1, 1 + len(meta.padded_spatial)))
            x = torch.roll(x, shifts=tuple(shift for _ in meta.padded_spatial), dims=dims)
        if any(meta.pad):
            slices = (slice(None),) + tuple(slice(0, size) for size in meta.spatial) + (slice(None),)
            x = x[slices].contiguous()
        return x


class GridWindowIndexer(WindowIndexer):
    """Grid attention grouper used by MaxViT. Paper: https://arxiv.org/abs/2204.01697"""

    def __init__(self, grid_size: int = 4):
        super().__init__()
        self.grid_size = int(grid_size)

    def _pad(self, x):
        spatial = x.shape[1:-1]
        pad = [(self.grid_size - (size % self.grid_size)) % self.grid_size for size in spatial]
        if any(pad):
            if len(spatial) == 2:
                x = F.pad(x, (0, 0, 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
        return x, pad

    def partition(self, x):
        spatial = tuple(x.shape[1:-1])
        x, pad = self._pad(x)
        padded_spatial = tuple(x.shape[1:-1])
        B, C = x.shape[0], x.shape[-1]
        nd = len(padded_spatial)
        block_shape = [size // self.grid_size for size in padded_spatial]
        view_shape = [B]
        for blocks in block_shape:
            view_shape.extend([blocks, self.grid_size])
        view_shape.append(C)
        x = x.view(*view_shape)
        permute = [0] + [2 + 2 * i for i in range(nd)] + [1 + 2 * i for i in range(nd)] + [2 * nd + 1]
        x = x.permute(*permute).contiguous()
        windows = x.view(B * (self.grid_size ** nd), -1, C)
        inverse = [0] * len(permute)
        for pos, orig in enumerate(permute):
            inverse[orig] = pos
        meta = WindowMeta(
            spatial=spatial,
            padded_spatial=padded_spatial,
            pad=pad,
            batch=B,
            view_shape=tuple(view_shape),
            permute=permute,
            inverse_permute=inverse,
            group_shape=(self.grid_size,) * nd,
            token_shape=tuple(block_shape),
        )
        return windows, meta

    def reverse(self, windows, meta: WindowMeta):
        C = windows.shape[-1]
        x = windows.view(meta.batch, *meta.group_shape, *meta.token_shape, C)
        x = x.permute(*meta.inverse_permute).contiguous().view(meta.batch, *meta.padded_spatial, C)
        if any(meta.pad):
            slices = (slice(None),) + tuple(slice(0, size) for size in meta.spatial) + (slice(None),)
            x = x[slices].contiguous()
        return x


class CrossWindowIndexer(WindowIndexer):
    """Axis stripe grouper inspired by CSWin. Paper: https://arxiv.org/abs/2107.00652"""

    def __init__(self, stripe_size: int = 4, axis: int = 0):
        super().__init__()
        self.stripe_size = int(stripe_size)
        self.axis = int(axis)

    def _pad(self, x):
        spatial = x.shape[1:-1]
        axis = self.axis % len(spatial)
        pad = [
            0 if i == axis else (self.stripe_size - (size % self.stripe_size)) % self.stripe_size
            for i, size in enumerate(spatial)
        ]
        if any(pad):
            if len(spatial) == 2:
                x = F.pad(x, (0, 0, 0, pad[1], 0, pad[0]))
            else:
                x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
        return x, pad

    def partition(self, x):
        spatial = tuple(x.shape[1:-1])
        x, pad = self._pad(x)
        padded_spatial = tuple(x.shape[1:-1])
        B, C = x.shape[0], x.shape[-1]
        nd = len(padded_spatial)
        axis = self.axis % nd

        view_shape = [B]
        dim_entries = []
        group_positions, token_positions = [], []
        cursor = 1
        group_shape, token_shape = [], []
        for i, size in enumerate(padded_spatial):
            if i == axis:
                view_shape.append(size)
                dim_entries.append((cursor,))
                token_positions.append(cursor)
                token_shape.append(size)
                cursor += 1
            else:
                chunks = size // self.stripe_size
                view_shape.extend([chunks, self.stripe_size])
                dim_entries.append((cursor, cursor + 1))
                group_positions.append(cursor)
                token_positions.append(cursor + 1)
                group_shape.append(chunks)
                token_shape.append(self.stripe_size)
                cursor += 2
        view_shape.append(C)
        channel_pos = cursor
        x = x.view(*view_shape)
        permute = [0] + group_positions + token_positions + [channel_pos]
        x = x.permute(*permute).contiguous()
        windows = x.view(B * max(1, int(torch.tensor(group_shape).prod().item()) if group_shape else 1), -1, C)
        inverse = [0] * len(permute)
        for pos, orig in enumerate(permute):
            inverse[orig] = pos
        meta = WindowMeta(
            spatial=spatial,
            padded_spatial=padded_spatial,
            pad=pad,
            batch=B,
            view_shape=tuple(view_shape),
            permute=permute,
            inverse_permute=inverse,
            group_shape=tuple(group_shape),
            token_shape=tuple(token_shape),
        )
        return windows, meta

    def reverse(self, windows, meta: WindowMeta):
        C = windows.shape[-1]
        x = windows.view(meta.batch, *meta.group_shape, *meta.token_shape, C)
        x = x.permute(*meta.inverse_permute).contiguous().view(meta.batch, *meta.padded_spatial, C)
        if any(meta.pad):
            slices = (slice(None),) + tuple(slice(0, size) for size in meta.spatial) + (slice(None),)
            x = x[slices].contiguous()
        return x


class SubsampledGlobalIndexer(WindowIndexer):
    """Subsampled global attention grouper used by Twins-SVT. Paper: https://arxiv.org/abs/2104.13840"""

    def __init__(self, sample_stride: int = 4):
        super().__init__()
        self.sample_stride = int(sample_stride)

    def partition(self, x):
        spatial = tuple(x.shape[1:-1])
        B, C = x.shape[0], x.shape[-1]
        nd = len(spatial)
        slices = (slice(None),) + tuple(slice(0, None, self.sample_stride) for _ in spatial) + (slice(None),)
        tokens = x[slices].contiguous().view(B, -1, C)
        meta = WindowMeta(
            spatial=spatial,
            padded_spatial=spatial,
            pad=[0] * nd,
            batch=B,
            view_shape=tuple(tokens.shape),
        )
        return tokens, meta

    def reverse(self, windows, meta: WindowMeta):
        coarse_shape = meta.view_shape[1]
        nd = len(meta.spatial)
        stride = self.sample_stride
        sampled_spatial = tuple((size + stride - 1) // stride for size in meta.spatial)
        C = windows.shape[-1]
        x = windows.view(meta.batch, *sampled_spatial, C)
        x = x.permute(0, nd + 1, *range(1, 1 + nd)).contiguous()
        mode = "bilinear" if nd == 2 else "trilinear"
        x = F.interpolate(x, size=meta.spatial, mode=mode, align_corners=False)
        return x.permute(0, *range(2, 2 + nd), 1).contiguous()


class FocalWindowIndexer(WindowIndexer):
    """Focal attention grouper with local fine tokens and coarse contextual tokens. Paper: https://arxiv.org/abs/2107.00641"""

    def __init__(self, window_size: int = 4, focal_stride: int = 2):
        super().__init__()
        self.local = SwinWindowIndexer(window_size=window_size, shift_size=0)
        self.focal_stride = int(focal_stride)

    def partition(self, x):
        local_windows, meta = self.local.partition(x)
        B, C = x.shape[0], x.shape[-1]
        nd = len(meta.spatial)
        x_cf = x.permute(0, nd + 1, *range(1, 1 + nd)).contiguous()
        if nd == 2:
            pooled = F.avg_pool2d(x_cf, kernel_size=self.focal_stride, stride=self.focal_stride, ceil_mode=True)
        else:
            pooled = F.avg_pool3d(x_cf, kernel_size=self.focal_stride, stride=self.focal_stride, ceil_mode=True)
        pooled = pooled.permute(0, *range(2, 2 + nd), 1).contiguous().view(B, -1, C)
        n_windows = local_windows.shape[0] // B
        pooled = pooled.unsqueeze(1).expand(B, n_windows, pooled.shape[1], C).reshape(B * n_windows, -1, C)
        meta.token_shape = (local_windows.shape[1],)
        return torch.cat([local_windows, pooled], dim=1), meta

    def reverse(self, windows, meta: WindowMeta):
        local_len = meta.token_shape[0]
        return self.local.reverse(windows[:, :local_len], meta)


class WindowAttention(nn.Module):
    """Window multi-head self-attention used by Swin. Paper: https://arxiv.org/abs/2103.14030"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"WindowAttention requires dim % num_heads == 0, got {dim}, {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2).to(attn.device, attn.dtype)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class WindowAttentionBlock(nn.Module):
    """Generic block: index tokens into groups, run attention, scatter back. Swin reference: https://arxiv.org/abs/2103.14030"""

    def __init__(self, dim: int, indexer: WindowIndexer, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.indexer = indexer
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        shortcut = x
        windows, meta = self.indexer.partition(self.norm1(x))
        x = shortcut + self.indexer.reverse(self.attn(windows, mask=meta.mask), meta)
        return x + self.mlp(self.norm2(x))


class AdditiveContextAttentionBlock(nn.Module):
    """Local attention plus additive sparse/global context. Twins reference: https://arxiv.org/abs/2104.13840"""

    def __init__(self, dim: int, local_indexer: WindowIndexer, context_indexer: WindowIndexer,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.local = WindowAttentionBlock(dim, local_indexer, num_heads, mlp_ratio, dropout)
        self.context_indexer = context_indexer
        self.norm_context = nn.LayerNorm(dim)
        self.context_attn = WindowAttention(dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x = self.local(x)
        windows, meta = self.context_indexer.partition(self.norm_context(x))
        return x + self.context_indexer.reverse(self.context_attn(windows, mask=meta.mask), meta)


class SwinBlock(WindowAttentionBlock):
    """Shifted-window Transformer block. Paper: https://arxiv.org/abs/2103.14030"""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, spatial_ndim: int = 2):
        super().__init__(
            dim,
            SwinWindowIndexer(window_size=window_size, shift_size=shift_size),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.spatial_ndim = spatial_ndim


class MaxViTBlock(nn.Module):
    """MaxViT-style block attention followed by grid attention. Paper: https://arxiv.org/abs/2204.01697"""

    def __init__(self, dim: int, num_heads: int = 4, block_size: int = 4,
                 grid_size: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.block_attn = WindowAttentionBlock(
            dim,
            SwinWindowIndexer(window_size=block_size, shift_size=0),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.grid_attn = WindowAttentionBlock(
            dim,
            GridWindowIndexer(grid_size=grid_size),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        return self.grid_attn(self.block_attn(x))


class CSWinBlock(nn.Module):
    """CSWin-style cross/axis window attention block. Paper: https://arxiv.org/abs/2107.00652"""

    def __init__(self, dim: int, num_heads: int = 4, stripe_size: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, spatial_ndim: int = 2,
                 axis: int = 0):
        super().__init__()
        self.block = WindowAttentionBlock(
            dim,
            CrossWindowIndexer(stripe_size=stripe_size, axis=axis % spatial_ndim),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        return self.block(x)


class TwinsBlock(nn.Module):
    """Twins-SVT style local grouped attention plus subsampled global attention. Paper: https://arxiv.org/abs/2104.13840"""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4,
                 sample_stride: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.block = AdditiveContextAttentionBlock(
            dim,
            SwinWindowIndexer(window_size=window_size, shift_size=0),
            SubsampledGlobalIndexer(sample_stride=sample_stride),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        return self.block(x)


class FocalTransformerBlock(WindowAttentionBlock):
    """Focal Transformer block using local fine tokens plus coarse contextual tokens. Paper: https://arxiv.org/abs/2107.00641"""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4,
                 focal_stride: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__(
            dim,
            FocalWindowIndexer(window_size=window_size, focal_stride=focal_stride),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )


class SwinEncoder(nn.Module):
    """Swin shifted-window encoder. Paper: https://arxiv.org/abs/2103.14030"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, window_size: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        if enc_ch % nhead != 0:
            raise ValueError(f"Swin requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        self.spatial_ndim = spatial_ndim
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        shift = max(0, window_size // 2)
        self.blocks = nn.ModuleList([
            SwinBlock(enc_ch, nhead, window_size, 0 if i % 2 == 0 else shift,
                      mlp_ratio, dropout, spatial_ndim)
            for i in range(depth)
        ])

    def forward(self, x):
        feat = self.stem(x)
        feat = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        return feat.permute(0, self.spatial_ndim + 1,
                            *range(1, 1 + self.spatial_ndim)).contiguous()


class MaxViTEncoder(nn.Module):
    """MaxViT encoder with block and grid attention. Paper: https://arxiv.org/abs/2204.01697"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, block_size: int = 4,
                 grid_size: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        if enc_ch % nhead != 0:
            raise ValueError(f"MaxViT requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        self.spatial_ndim = spatial_ndim
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.ModuleList([
            MaxViTBlock(enc_ch, nhead, block_size, grid_size, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        feat = self.stem(x)
        feat = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        return feat.permute(0, self.spatial_ndim + 1,
                            *range(1, 1 + self.spatial_ndim)).contiguous()


class CSWinEncoder(nn.Module):
    """CSWin encoder with alternating axis stripe attention. Paper: https://arxiv.org/abs/2107.00652"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, stripe_size: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        if enc_ch % nhead != 0:
            raise ValueError(f"CSWin requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        self.spatial_ndim = spatial_ndim
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.ModuleList([
            CSWinBlock(enc_ch, nhead, stripe_size, mlp_ratio, dropout, spatial_ndim, axis=i % spatial_ndim)
            for i in range(depth)
        ])

    def forward(self, x):
        feat = self.stem(x)
        feat = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        return feat.permute(0, self.spatial_ndim + 1,
                            *range(1, 1 + self.spatial_ndim)).contiguous()


class TwinsEncoder(nn.Module):
    """Twins-SVT encoder with local grouped and subsampled global attention. Paper: https://arxiv.org/abs/2104.13840"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, window_size: int = 4,
                 sample_stride: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        if enc_ch % nhead != 0:
            raise ValueError(f"Twins requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        self.spatial_ndim = spatial_ndim
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.ModuleList([
            TwinsBlock(enc_ch, nhead, window_size, sample_stride, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        feat = self.stem(x)
        feat = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        return feat.permute(0, self.spatial_ndim + 1,
                            *range(1, 1 + self.spatial_ndim)).contiguous()


class FocalTransformerEncoder(nn.Module):
    """Focal Transformer encoder with local and coarse contextual tokens. Paper: https://arxiv.org/abs/2107.00641"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, window_size: int = 4,
                 focal_stride: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__()
        if enc_ch % nhead != 0:
            raise ValueError(f"FocalTransformer requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        self.spatial_ndim = spatial_ndim
        self.out_ch = enc_ch
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(enc_ch, nhead, window_size, focal_stride, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        feat = self.stem(x)
        feat = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        return feat.permute(0, self.spatial_ndim + 1,
                            *range(1, 1 + self.spatial_ndim)).contiguous()


class WindowAttentionModel(GridEncoderMLPModel):
    """Window-attention encoder plus shared MLP decoder. Swin reference: https://arxiv.org/abs/2103.14030"""

    ENCODERS = {
        "Swin": SwinEncoder,
        "MaxViT": MaxViTEncoder,
        "CSWin": CSWinEncoder,
        "Twins": TwinsEncoder,
        "FocalTransformer": FocalTransformerEncoder,
    }

    def __init__(self, n_grad: int, spatial_dim: int = 3, out_params: int = 7,
                 encoder: dict | None = None, decoder: dict | None = None,
                 use_KAN: bool = False, use_VAE: bool = False,
                 latent_dim: int = 64, vae_hidden: int = 128):
        encoder = encoder or {"type": "Swin"}
        decoder = decoder or {}
        enc_type = encoder.get("type", "Swin")
        if enc_type not in self.ENCODERS:
            raise ValueError(f"Unknown window-attention encoder type {enc_type!r}. Options: {tuple(self.ENCODERS)}")
        enc_cls = self.ENCODERS[enc_type]
        enc_kwargs = filter_init_kwargs(enc_cls, {k: v for k, v in encoder.items() if k != "type"})
        enc = enc_cls(
            n_grad=n_grad,
            spatial_ndim=spatial_dim,
            use_kan=use_KAN,
            **enc_kwargs,
        )
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


class Swin(WindowAttentionModel):
    """Swin encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2103.14030"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "Swin", **(encoder or {})}, **kwargs)


class MaxViT(WindowAttentionModel):
    """MaxViT encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2204.01697"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "MaxViT", **(encoder or {})}, **kwargs)


class CSWin(WindowAttentionModel):
    """CSWin encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2107.00652"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "CSWin", **(encoder or {})}, **kwargs)


class Twins(WindowAttentionModel):
    """Twins-SVT encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2104.13840"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "Twins", **(encoder or {})}, **kwargs)


class FocalTransformer(WindowAttentionModel):
    """Focal Transformer encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/2107.00641"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "FocalTransformer", **(encoder or {})}, **kwargs)
