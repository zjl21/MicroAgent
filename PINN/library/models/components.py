import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoGroupNorm(nn.Module):
    """GroupNorm wrapper. Paper: https://arxiv.org/abs/1803.08494"""

    def __init__(self, channels: int, num_groups: int = 8, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels, eps=eps, affine=affine)

    def forward(self, x):
        return self.norm(x)


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network linear layer. Paper: https://arxiv.org/abs/2404.19756"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation=nn.GELU,
        grid_range=(-1.0, 1.0),
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = tuple(grid_range)
        self.norm = norm_layer(output_dim, **norm_kwargs) if norm_layer is not None else nn.Identity()
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.base_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.spline_linear = nn.Linear((grid_size + spline_order) * input_dim, output_dim, bias=False)

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.base_linear.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.spline_linear.weight, nonlinearity="linear")

    def _bases(self, x):
        x_uns = x.unsqueeze(-1)
        grid = self.grid.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).contiguous().to(x.device)
        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        return bases.contiguous().flatten(1, 2)

    def forward(self, x):
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).float()
        base_output = self.base_linear(self.base_activation(x))
        spline_output = self.spline_linear(self._bases(x))
        y = self.dropout(self.act(self.norm(base_output + spline_output)))
        return y.reshape(*orig_shape, self.outdim)


class KANMLP(nn.Module):
    """KAN MLP decoder. Paper: https://arxiv.org/abs/2404.19756"""

    def __init__(
        self,
        in_ch: int,
        hidden: int,
        num_layers: int,
        out_params: int,
        neck_ch: int = 64,
        dropout: float = 0.0,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation=nn.GELU,
        grid_range=(-1.0, 1.0),
        **kwargs,
    ):
        super().__init__()
        self.out_params = out_params
        self.hidden_layers = nn.ModuleList()
        ch = in_ch
        for _ in range(num_layers):
            self.hidden_layers.append(
                KANLinear(
                    ch,
                    hidden,
                    spline_order=spline_order,
                    grid_size=grid_size,
                    base_activation=base_activation,
                    grid_range=grid_range,
                    dropout=dropout,
                )
            )
            ch = hidden
        self.neck = KANLinear(
            hidden,
            neck_ch,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
        )
        self.head = nn.Linear(neck_ch, out_params)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = x.float()
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.neck(x)
        return self.head(x)


class KANConvNDLayer(nn.Module):
    """N-D KAN convolution layer. KAN paper: https://arxiv.org/abs/2404.19756"""

    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0,
                 **norm_kwargs):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = tuple(grid_range)
        self.norm_kwargs = norm_kwargs

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if input_dim % groups != 0:
            raise ValueError("input_dim must be divisible by groups")
        if output_dim % groups != 0:
            raise ValueError("output_dim must be divisible by groups")

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class((grid_size + spline_order) * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32,
        )
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity="linear")
        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity="linear")

    def forward_kan(self, x, group_index):
        base_output = self.base_conv[group_index](self.base_activation(x))
        x_uns = x.unsqueeze(-1)
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*([1 for _ in range(self.ndim + 1)] + [-1])).expand(target).contiguous().to(x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            output.append(self.forward_kan(_x, group_ind).clone())
        return torch.cat(output, dim=1)


class KANConv3DLayer(KANConvNDLayer):
    """3D KAN convolution layer. KAN paper: https://arxiv.org/abs/2404.19756"""

    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super().__init__(nn.Conv3d, norm_layer,
                         input_dim, output_dim,
                         spline_order, kernel_size,
                         groups=groups, padding=padding, stride=stride, dilation=dilation,
                         ndim=3,
                         grid_size=grid_size, base_activation=base_activation,
                         grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv2DLayer(KANConvNDLayer):
    """2D KAN convolution layer. KAN paper: https://arxiv.org/abs/2404.19756"""

    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super().__init__(nn.Conv2d, norm_layer,
                         input_dim, output_dim,
                         spline_order, kernel_size,
                         groups=groups, padding=padding, stride=stride, dilation=dilation,
                         ndim=2,
                         grid_size=grid_size, base_activation=base_activation,
                         grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv1DLayer(KANConvNDLayer):
    """1D KAN convolution layer. KAN paper: https://arxiv.org/abs/2404.19756"""

    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=(-1, 1), dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super().__init__(nn.Conv1d, norm_layer,
                         input_dim, output_dim,
                         spline_order, kernel_size,
                         groups=groups, padding=padding, stride=stride, dilation=dilation,
                         ndim=1,
                         grid_size=grid_size, base_activation=base_activation,
                         grid_range=grid_range, dropout=dropout, **norm_kwargs)


class _ChannelAttention(nn.Module):
    """CBAM channel attention submodule. Paper: https://arxiv.org/abs/1807.06521"""

    def __init__(self, channels: int, reduction: int = 16, spatial_ndim: int = 2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        pool_cls = nn.AdaptiveAvgPool2d if spatial_ndim == 2 else nn.AdaptiveAvgPool3d
        max_cls = nn.AdaptiveMaxPool2d if spatial_ndim == 2 else nn.AdaptiveMaxPool3d
        self.avg_pool = pool_cls(1)
        self.max_pool = max_cls(1)

    def forward(self, x):
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        return x * w.view(shape)


class _SpatialAttention(nn.Module):
    """CBAM spatial attention submodule. Paper: https://arxiv.org/abs/1807.06521"""

    def __init__(self, spatial_ndim: int = 2, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        conv_cls = nn.Conv2d if spatial_ndim == 2 else nn.Conv3d
        self.conv = conv_cls(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module. Paper: https://arxiv.org/abs/1807.06521"""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_ndim: int = 2, kernel_size: int = 7,
                 use_channel_attn: bool = True, use_spatial_attn: bool = True):
        super().__init__()
        self.channel_attn = (
            _ChannelAttention(channels, reduction, spatial_ndim)
            if use_channel_attn else nn.Identity()
        )
        self.spatial_attn = (
            _SpatialAttention(spatial_ndim, kernel_size)
            if use_spatial_attn else nn.Identity()
        )

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class FeatureGaussianBottleneck(nn.Module):
    """VAE-style Gaussian bottleneck. Paper: https://arxiv.org/abs/1312.6114"""

    def __init__(self, in_dim: int, latent_dim: int = 64, hidden: int = 128, out_dim: int | None = None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc_out = nn.Linear(latent_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc_mu.weight, std=0.01)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.normal_(self.fc_logvar.weight, std=0.01)
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.kaiming_normal_(self.fc_out.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        h = self.act(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-12.0, 12.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        y = self.act(self.fc_out(z))
        kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1))
        return y, mu, logvar, kl


class VoxelMLP(nn.Module):
    """Per-voxel MLP decoder. MLP/backprop reference: https://www.nature.com/articles/323533a0"""

    def __init__(self, in_ch: int, hidden: int, num_layers: int, out_params: int,
                 neck_ch: int = 64, dropout: float = 0.0):
        super().__init__()
        self.out_params = out_params

        layers = []
        ch = in_ch
        for _ in range(num_layers):
            fc = nn.Linear(ch, hidden)
            nn.init.kaiming_normal_(fc.weight, a=0.0, nonlinearity="relu", mode="fan_in")
            nn.init.zeros_(fc.bias)
            layers.append(fc)
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            ch = hidden
        self.hidden_layers = nn.Sequential(*layers)

        self.neck = nn.Linear(hidden, neck_ch)
        nn.init.kaiming_normal_(self.neck.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        nn.init.zeros_(self.neck.bias)

        self.head = nn.Linear(neck_ch, out_params)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.neck(x)
        return self.head(x)


def make_voxel_mlp(in_ch, hidden, num_layers, out_params, neck_ch, dropout, use_kan=False):
    if use_kan:
        return KANMLP(in_ch, hidden, num_layers, out_params, neck_ch=neck_ch, dropout=dropout)
    return VoxelMLP(in_ch, hidden, num_layers, out_params, neck_ch=neck_ch, dropout=dropout)


def filter_init_kwargs(cls, cfg):
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in cfg.items() if k in allowed}


class EncoderMLPModel(nn.Module):
    """Shared encoder + optional Gaussian bottleneck + MLP decoder. VAE: https://arxiv.org/abs/1312.6114"""

    def __init__(self, encoder: nn.Module, mlp: nn.Module,
                 enc_ch: int, use_VAE: bool = False,
                 latent_dim: int = 64, vae_hidden: int = 128):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.enc_ch = enc_ch
        self.use_VAE = bool(use_VAE)
        self.kl_loss = torch.tensor(0.0)
        self.mu = None
        self.logvar = None
        self.bottleneck = (
            FeatureGaussianBottleneck(enc_ch, latent_dim=latent_dim, hidden=vae_hidden, out_dim=enc_ch)
            if self.use_VAE else nn.Identity()
        )

    def _apply_bottleneck(self, features):
        if self.use_VAE:
            features, mu, logvar, kl = self.bottleneck(features)
            self.mu = mu
            self.logvar = logvar
            self.kl_loss = kl
        else:
            self.mu = None
            self.logvar = None
            self.kl_loss = features.sum() * 0.0
        return features

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError


class GridEncoderMLPModel(EncoderMLPModel):
    """Encoder + Gaussian bottleneck + per-voxel MLP decoder for 2D/3D grids."""

    def __init__(self, encoder: nn.Module, mlp: nn.Module,
                 enc_ch: int, spatial_ndim: int,
                 use_VAE: bool = False, latent_dim: int = 64, vae_hidden: int = 128):
        super().__init__(
            encoder,
            mlp,
            enc_ch=enc_ch,
            use_VAE=use_VAE,
            latent_dim=latent_dim,
            vae_hidden=vae_hidden,
        )
        self.spatial_ndim = spatial_ndim

    def forward(self, x, mask=None):
        nd = self.spatial_ndim
        if x.ndim - 2 != nd:
            raise ValueError(
                f"{self.__class__.__name__} expects {nd}D spatial input "
                f"(ndim={nd + 2}), got ndim={x.ndim}."
            )

        B = x.shape[0]
        S = x.shape[2:]
        P = self.mlp.out_params

        feat = self.encoder(x)
        if feat.shape[2:] != S:
            mode = "bilinear" if nd == 2 else "trilinear"
            feat = F.interpolate(feat, size=S, mode=mode, align_corners=False)

        feat_cl = feat.permute(0, *range(2, 2 + nd), 1).contiguous()
        feat_flat = feat_cl.reshape(-1, self.enc_ch)
        feat_latent = self._apply_bottleneck(feat_flat)
        feat_cl = feat_latent.reshape(*feat_cl.shape[:-1], self.enc_ch)

        out = torch.zeros(*feat_cl.shape[:-1], P, device=x.device, dtype=x.dtype)
        if mask is not None:
            mask_bool = mask.bool()
            if mask_bool.ndim == nd + 2 and mask_bool.shape[1] == 1:
                mask_bool = mask_bool.squeeze(1)
            if mask_bool.shape != feat_cl.shape[:-1]:
                raise ValueError(
                    f"mask shape must be (B, *S) or (B, 1, *S), got {tuple(mask.shape)} "
                    f"for feature shape {tuple(feat_cl.shape[:-1])}."
                )
            out[mask_bool] = self.mlp(feat_cl[mask_bool])
        else:
            out = self.mlp(feat_cl.reshape(-1, self.enc_ch)).reshape(B, *S, P)
        return out.permute(0, nd + 1, *range(1, nd + 1)).contiguous()


class PointEncoderMLPModel(EncoderMLPModel):
    """Encoder + Gaussian bottleneck + MLP decoder for coordinate/token point sets."""

    def forward(self, x):
        features = self.encoder(x)
        if features.shape[-1] != self.enc_ch:
            raise ValueError(
                f"{self.__class__.__name__} encoder must return last dim {self.enc_ch}, "
                f"got {features.shape[-1]}."
            )
        flat = features.reshape(-1, self.enc_ch)
        latent = self._apply_bottleneck(flat).reshape_as(features)
        return self.mlp(latent)


class ConvResidualBlock(nn.Module):
    """2D/3D Conv or KANConv block with optional residual projection. ResNet: https://arxiv.org/abs/1512.03385"""

    def __init__(self, in_ch: int, out_ch: int, spatial_ndim: int = 2,
                 use_gn: bool = False, use_kan: bool = False,
                 residual: bool = False, dilation: int = 1, stride: int = 1,
                 num_convs: int = 1):
        super().__init__()
        if spatial_ndim not in (2, 3):
            raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        layers = []
        ch = in_ch
        for i in range(num_convs):
            conv_stride = stride if i == 0 else 1
            if use_kan:
                conv_cls = KANConv2DLayer if spatial_ndim == 2 else KANConv3DLayer
                if use_gn:
                    norm_layer = lambda channels, **kwargs: nn.GroupNorm(min(8, channels), channels)
                else:
                    norm_layer = lambda channels, **kwargs: nn.Identity()
                layers.append(
                    conv_cls(
                        ch, out_ch, kernel_size=3, padding=dilation,
                        stride=conv_stride, dilation=dilation, norm_layer=norm_layer,
                    )
                )
            else:
                conv_cls = nn.Conv2d if spatial_ndim == 2 else nn.Conv3d
                conv = conv_cls(ch, out_ch, kernel_size=3, padding=dilation,
                                stride=conv_stride, dilation=dilation)
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                nn.init.zeros_(conv.bias)
                norm = nn.GroupNorm(min(8, out_ch), out_ch) if use_gn else nn.Identity()
                layers.extend([conv, norm, nn.ReLU(inplace=True)])
            ch = out_ch
        self.main = nn.Sequential(*layers)

        self.proj = None
        if residual:
            if in_ch != out_ch or stride != 1:
                proj_cls = nn.Conv2d if spatial_ndim == 2 else nn.Conv3d
                self.proj = proj_cls(in_ch, out_ch, kernel_size=1, stride=stride)
                nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
                nn.init.zeros_(self.proj.bias)
            else:
                self.proj = nn.Identity()

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            skip = self.proj(x)
            if skip.shape[2:] != y.shape[2:]:
                mode = "bilinear" if y.ndim == 4 else "trilinear"
                skip = F.interpolate(skip, size=y.shape[2:], mode=mode, align_corners=False)
            y = y + skip
        return y


def make_conv_stem(n_grad: int, enc_ch: int, spatial_ndim: int,
                   stem_layers: int = 1, use_kan: bool = False):
    layers = []
    in_ch = n_grad
    for _ in range(max(1, stem_layers)):
        if use_kan:
            conv_cls = KANConv2DLayer if spatial_ndim == 2 else KANConv3DLayer
            layers.append(conv_cls(in_ch, enc_ch, kernel_size=3, padding=1))
        else:
            conv_cls = nn.Conv2d if spatial_ndim == 2 else nn.Conv3d
            conv = conv_cls(in_ch, enc_ch, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_ch = enc_ch
    return nn.Sequential(*layers)
