import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import CBAM, ConvResidualBlock, GridEncoderMLPModel, filter_init_kwargs, make_voxel_mlp


def _attention_kwargs(attention, spatial_ndim):
    if not attention or not attention.get("enabled", True):
        return None
    return dict(
        reduction=attention.get("reduction", 16),
        kernel_size=attention.get("kernel_size", 7),
        spatial_ndim=spatial_ndim,
        use_channel_attn=attention.get("use_channel_attn", True),
        use_spatial_attn=attention.get("use_spatial_attn", True),
    )


class CNNEncoder(nn.Module):
    """CNN spatial encoder. CNN reference: https://ieeexplore.ieee.org/document/726791"""

    def __init__(self, n_grad: int, spatial_ndim: int,
                 enc_ch: int = 32, enc_layers: int = 1,
                 residual: bool = False, dilation: int = 1, stride: int = 1,
                 attention: dict | None = None, use_kan: bool = False):
        super().__init__()
        self.spatial_ndim = spatial_ndim
        attn_kw = _attention_kwargs(attention, spatial_ndim)
        layers = []
        in_ch = n_grad
        for _ in range(enc_layers):
            block = [
                ConvResidualBlock(
                    in_ch,
                    enc_ch,
                    spatial_ndim=spatial_ndim,
                    use_kan=use_kan,
                    residual=residual,
                    dilation=dilation,
                    stride=stride,
                    num_convs=1,
                )
            ]
            if attn_kw:
                block.append(CBAM(enc_ch, **attn_kw))
            layers.append(nn.Sequential(*block))
            in_ch = enc_ch
        self.layers = nn.Sequential(*layers)
        self.out_ch = enc_ch

    def forward(self, x):
        target_shape = x.shape[2:]
        x = self.layers(x)
        if x.shape[2:] != target_shape:
            mode = "bilinear" if self.spatial_ndim == 2 else "trilinear"
            x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
        return x


class UNetEncoder(nn.Module):
    """U-Net spatial encoder. Paper: https://arxiv.org/abs/1505.04597"""

    def __init__(self, n_grad: int, spatial_ndim: int,
                 hidden: int = 64, num_layers: int = 3, use_gn: bool = False,
                 residual: bool = False, dilation: int = 1, stride: int = 1,
                 attention: dict | None = None, use_kan: bool = False):
        super().__init__()
        self.spatial_ndim = spatial_ndim
        attn_kw = _attention_kwargs(attention, spatial_ndim)
        pool_cls = nn.MaxPool2d if spatial_ndim == 2 else nn.MaxPool3d
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        enc_channels = []

        in_ch, ch = n_grad, hidden
        for _ in range(num_layers):
            self.encoders.append(self._block(in_ch, ch, use_gn, use_kan, residual, dilation, stride))
            self.pools.append(pool_cls(2))
            self.enc_attns.append(CBAM(ch, **attn_kw) if attn_kw else nn.Identity())
            enc_channels.append(ch)
            in_ch, ch = ch, ch * 2

        self.bottleneck = self._block(in_ch, ch, use_gn, use_kan, residual, dilation, 1)
        self.btn_attn = CBAM(ch, **attn_kw) if attn_kw else nn.Identity()
        self.decoders = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        for enc_ch in reversed(enc_channels):
            self.decoders.append(self._block(ch + enc_ch, enc_ch, use_gn, use_kan, residual, dilation, 1))
            self.dec_attns.append(CBAM(enc_ch, **attn_kw) if attn_kw else nn.Identity())
            ch = enc_ch
        self.out_ch = hidden

    def _block(self, in_ch, out_ch, use_gn, use_kan, residual, dilation, stride):
        return ConvResidualBlock(
            in_ch,
            out_ch,
            spatial_ndim=self.spatial_ndim,
            use_gn=use_gn,
            use_kan=use_kan,
            residual=residual,
            dilation=dilation,
            stride=stride,
            num_convs=2,
        )

    def forward(self, x):
        target_shape = x.shape[2:]
        mode = "bilinear" if self.spatial_ndim == 2 else "trilinear"
        skips, feat = [], x
        for enc, pool, attn in zip(self.encoders, self.pools, self.enc_attns):
            feat = attn(enc(feat))
            skips.append(feat)
            feat = pool(feat)

        feat = self.btn_attn(self.bottleneck(feat))
        for dec, attn, skip in zip(self.decoders, self.dec_attns, reversed(skips)):
            feat = F.interpolate(feat, size=skip.shape[2:], mode=mode, align_corners=False)
            feat = attn(dec(torch.cat([feat, skip], dim=1)))

        if feat.shape[2:] != target_shape:
            feat = F.interpolate(feat, size=target_shape, mode=mode, align_corners=False)
        return feat


class SpatialModel(GridEncoderMLPModel):
    """Direct spatial encoder plus shared MLP decoder. CNN: https://ieeexplore.ieee.org/document/726791 ; U-Net: https://arxiv.org/abs/1505.04597"""

    ENCODERS = {
        "CNN": CNNEncoder,
        "UNet": UNetEncoder,
    }

    def __init__(self, n_grad: int, spatial_dim: int = 3, out_params: int = 7,
                 encoder: dict | None = None, decoder: dict | None = None,
                 use_KAN: bool = False, use_VAE: bool = False,
                 latent_dim: int = 64, vae_hidden: int = 128):
        encoder = encoder or {"type": "CNN"}
        decoder = decoder or {}
        enc_type = encoder.get("type", "CNN")
        if enc_type not in self.ENCODERS:
            raise ValueError(f"Unknown spatial encoder type {enc_type!r}. Options: {tuple(self.ENCODERS)}")
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
        super().__init__(
            enc,
            mlp,
            enc_ch=enc.out_ch,
            spatial_ndim=spatial_dim,
            use_VAE=use_VAE,
            latent_dim=latent_dim,
            vae_hidden=vae_hidden,
        )


class CNN(SpatialModel):
    """CNN encoder plus shared MLP decoder. CNN reference: https://ieeexplore.ieee.org/document/726791"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        encoder = {"type": "CNN", **(encoder or {})}
        super().__init__(*args, encoder=encoder, **kwargs)


class UNet(SpatialModel):
    """U-Net encoder plus shared MLP decoder. Paper: https://arxiv.org/abs/1505.04597"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        encoder = {"type": "UNet", **(encoder or {})}
        super().__init__(*args, encoder=encoder, **kwargs)
