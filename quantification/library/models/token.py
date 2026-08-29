import torch
import torch.nn as nn

from .components import GridEncoderMLPModel, filter_init_kwargs, make_conv_stem, make_voxel_mlp


class SequenceTokenEncoder(nn.Module):
    """Conv stem plus spatial-to-token scanner. Transformer tokenization reference: https://arxiv.org/abs/1706.03762"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 scan_mode: str = "bidirectional", stem_layers: int = 1,
                 use_kan: bool = False):
        super().__init__()
        if spatial_ndim not in (2, 3):
            raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")
        scan_mode = (scan_mode or "bidirectional").lower()
        if scan_mode not in ("flatten", "bidirectional", "axial"):
            raise ValueError(f"scan_mode must be flatten, bidirectional, or axial, got {scan_mode!r}.")
        self.out_ch = enc_ch
        self.spatial_ndim = spatial_ndim
        self.scan_mode = scan_mode
        self.stem = make_conv_stem(n_grad, enc_ch, spatial_ndim, stem_layers=stem_layers, use_kan=use_kan)
        self.out_norm = nn.LayerNorm(enc_ch)

    def _mix_tokens(self, tokens):
        raise NotImplementedError

    def _run_flatten(self, feat, bidirectional: bool = False):
        B, C = feat.shape[:2]
        S = feat.shape[2:]
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        if bidirectional:
            fwd = self._mix_tokens(tokens)
            bwd = torch.flip(self._mix_tokens(torch.flip(tokens, dims=[1])), dims=[1])
            tokens = 0.5 * (fwd + bwd)
        else:
            tokens = self._mix_tokens(tokens)
        tokens = self.out_norm(tokens)
        return tokens.transpose(1, 2).reshape(B, C, *S).contiguous()

    def _run_axis(self, feat_cl, axis: int):
        nd = self.spatial_ndim
        B = feat_cl.shape[0]
        C = feat_cl.shape[-1]
        spatial = feat_cl.shape[1:-1]
        order = [0, axis + 1] + [i + 1 for i in range(nd) if i != axis] + [nd + 1]
        seq = feat_cl.permute(*order).contiguous()
        L = spatial[axis]
        seq = seq.reshape(-1, L, C)
        seq = self.out_norm(self._mix_tokens(seq))
        other = [spatial[i] for i in range(nd) if i != axis]
        seq = seq.reshape(B, L, *other, C)
        inv_order = [0] * (nd + 2)
        for pos, orig in enumerate(order):
            inv_order[orig] = pos
        return seq.permute(*inv_order).contiguous()

    def _run_axial(self, feat):
        feat_cl = feat.permute(0, *range(2, 2 + self.spatial_ndim), 1).contiguous()
        mixed = 0.0
        for axis in range(self.spatial_ndim):
            mixed = mixed + self._run_axis(feat_cl, axis)
        mixed = mixed / float(self.spatial_ndim)
        return mixed.permute(0, self.spatial_ndim + 1,
                             *range(1, self.spatial_ndim + 1)).contiguous()

    def forward(self, x):
        feat = self.stem(x)
        if self.scan_mode == "axial":
            return self._run_axial(feat)
        return self._run_flatten(feat, bidirectional=(self.scan_mode == "bidirectional"))


class MambaBlock(nn.Module):
    """Mamba selective state-space block. Paper: https://arxiv.org/abs/2312.00752"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0,
                 variant: str = "mamba", headdim: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        variant = (variant or "mamba").lower()
        if variant == "mamba":
            from mamba_ssm import Mamba as MambaSSM
            self.mixer = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                  expand=expand, use_fast_path=True)
        elif variant == "mamba2":
            from mamba_ssm import Mamba2 as MambaSSM
            self.mixer = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                  expand=expand, headdim=headdim, use_mem_eff_path=True)
        else:
            raise ValueError(f"Unsupported Mamba variant: {variant!r}")

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


class MambaEncoder(SequenceTokenEncoder):
    """Mamba token encoder. Paper: https://arxiv.org/abs/2312.00752"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64, depth: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.0, scan_mode: str = "bidirectional",
                 stem_layers: int = 1, variant: str = "mamba", headdim: int = 16,
                 use_kan: bool = False):
        super().__init__(n_grad, spatial_ndim, enc_ch, scan_mode, stem_layers, use_kan)
        self.blocks = nn.ModuleList([
            MambaBlock(enc_ch, d_state=d_state, d_conv=d_conv, expand=expand,
                       dropout=dropout, variant=variant, headdim=headdim)
            for _ in range(depth)
        ])

    def _mix_tokens(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class RecurrentEncoder(SequenceTokenEncoder):
    """RNN/LSTM token encoder. RNN: https://www.nature.com/articles/323533a0 ; LSTM: https://doi.org/10.1162/neco.1997.9.8.1735"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, dropout: float = 0.0,
                 scan_mode: str = "bidirectional", stem_layers: int = 1,
                 use_kan: bool = False, cell: str = "rnn"):
        super().__init__(n_grad, spatial_ndim, enc_ch, scan_mode, stem_layers, use_kan)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(enc_ch, enc_ch, num_layers=depth,
                           dropout=dropout if depth > 1 else 0.0,
                           batch_first=True, bidirectional=False)

    def _mix_tokens(self, tokens):
        y, _ = self.rnn(tokens)
        return y


class GlobalTransformerEncoder(SequenceTokenEncoder):
    """Global Transformer token encoder. Paper: https://arxiv.org/abs/1706.03762"""

    def __init__(self, n_grad: int, spatial_ndim: int, enc_ch: int = 64,
                 depth: int = 2, nhead: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.0, scan_mode: str = "bidirectional",
                 stem_layers: int = 1, use_kan: bool = False):
        super().__init__(n_grad, spatial_ndim, enc_ch, scan_mode, stem_layers, use_kan)
        if enc_ch % nhead != 0:
            raise ValueError(f"GlobalTransformer requires enc_ch % nhead == 0, got {enc_ch}, {nhead}.")
        layer = nn.TransformerEncoderLayer(
            d_model=enc_ch, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def _mix_tokens(self, tokens):
        return self.encoder(tokens)


class TokenModel(GridEncoderMLPModel):
    """Token encoder plus shared MLP decoder. Transformer reference: https://arxiv.org/abs/1706.03762"""

    ENCODERS = {
        "Mamba": MambaEncoder,
        "RNN": RecurrentEncoder,
        "LSTM": RecurrentEncoder,
        "Transformer": GlobalTransformerEncoder,
    }

    def __init__(self, n_grad: int, spatial_dim: int = 3, out_params: int = 7,
                 encoder: dict | None = None, decoder: dict | None = None,
                 use_KAN: bool = False, use_VAE: bool = False,
                 latent_dim: int = 64, vae_hidden: int = 128):
        encoder = encoder or {"type": "Mamba"}
        decoder = decoder or {}
        enc_type = encoder.get("type", "Mamba")
        if enc_type not in self.ENCODERS:
            raise ValueError(f"Unknown token encoder type {enc_type!r}. Options: {tuple(self.ENCODERS)}")
        enc_cls = self.ENCODERS[enc_type]
        enc_kwargs = filter_init_kwargs(enc_cls, {k: v for k, v in encoder.items() if k != "type"})
        if enc_type in ("RNN", "LSTM"):
            enc_kwargs["cell"] = enc_type.lower()
        enc = enc_cls(
            n_grad=n_grad, spatial_ndim=spatial_dim, use_kan=use_KAN, **enc_kwargs
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


class Mamba(TokenModel):
    """Mamba token model. Paper: https://arxiv.org/abs/2312.00752"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "Mamba", **(encoder or {})}, **kwargs)


class RNN(TokenModel):
    """RNN token model. RNN reference: https://www.nature.com/articles/323533a0"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "RNN", **(encoder or {})}, **kwargs)


class LSTM(TokenModel):
    """LSTM token model. Paper: https://doi.org/10.1162/neco.1997.9.8.1735"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "LSTM", **(encoder or {})}, **kwargs)


class Transformer(TokenModel):
    """Global Transformer token model. Paper: https://arxiv.org/abs/1706.03762"""

    def __init__(self, *args, encoder: dict | None = None, **kwargs):
        super().__init__(*args, encoder={"type": "Transformer", **(encoder or {})}, **kwargs)
