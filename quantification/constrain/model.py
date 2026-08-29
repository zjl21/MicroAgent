MODEL_FAMILIES = ("voxelwise", "coords", "spatial", "token", "window_attention", "operator")
SPATIAL_ENCODERS = ("CNN", "UNet")
TOKEN_ENCODERS = ("Mamba", "RNN", "LSTM", "Transformer")
WINDOW_ENCODERS = ("Swin", "MaxViT", "CSWin", "Twins", "FocalTransformer")
OPERATOR_ENCODERS = ("FNO", "AFNO", "UFNO", "UNO", "U-FNO")
ENCODER_FAMILIES = ("spatial", "token", "window_attention", "operator")
HASH_GRID_N_FEATURES_PER_LEVEL = (1, 2, 4, 8)


MODEL_CONSTRAINTS = [
    {
        "name": "model family is valid",
        "when": "always",
        "require": "model.name is one of voxelwise, coords, spatial, token, window_attention, operator",
        "applies": lambda config: True,
        "valid": lambda config: _model_family(config) in MODEL_FAMILIES,
        "message": lambda config: (
            f"model.name must be one of {MODEL_FAMILIES}, got {_model_family(config)!r}."
        ),
    },
    {
        "name": "model family block exists",
        "when": "always",
        "require": "model.<name> is a dictionary",
        "applies": lambda config: _model_family(config) in MODEL_FAMILIES,
        "valid": lambda config: isinstance(_model_cfg(config).get(_model_family(config)), dict),
        "message": lambda config: f"model.{_model_family(config)} must be a dictionary.",
    },
    {
        "name": "image-to-grid families need spatial data",
        "when": "model.name in ['spatial', 'token', 'window_attention', 'operator']",
        "require": "data.dataset in ['slicewise', 'patchwise']",
        "applies": lambda config: _model_family(config) in ENCODER_FAMILIES,
        "valid": lambda config: _dataset_type(config) in ("slicewise", "patchwise"),
        "message": lambda config: (
            f"model.name='{_model_family(config)}' requires slicewise or patchwise data, "
            f"not {_dataset_type(config)}."
        ),
    },
    {
        "name": "voxelwise family needs voxel data",
        "when": "model.name == 'voxelwise'",
        "require": "data.dataset == 'voxelwise'",
        "applies": lambda config: _model_family(config) == "voxelwise",
        "valid": lambda config: _dataset_type(config) == "voxelwise",
        "message": lambda config: f"model.name='voxelwise' requires voxelwise data, not {_dataset_type(config)}.",
    },
    {
        "name": "coordinate family needs coordinate data",
        "when": "model.name == 'coords'",
        "require": "data.dataset in ['3Dcoord', 'coords']",
        "applies": lambda config: _model_family(config) == "coords",
        "valid": lambda config: _dataset_type(config) in ("3Dcoord", "coords"),
        "message": lambda config: (
            f"model.name='coords' requires data.dataset='3Dcoord' or 'coords', "
            f"not {_dataset_type(config)}."
        ),
    },
    {
        "name": "encoder decoder model config shape is valid",
        "when": "model.name in ['spatial', 'token', 'window_attention', 'operator']",
        "require": "model.<family>.encoder and model.<family>.decoder are dictionaries",
        "applies": lambda config: _model_family(config) in ENCODER_FAMILIES,
        "valid": lambda config: _valid_encoder_decoder_cfg_shape(config),
        "message": lambda config: _encoder_decoder_cfg_shape_message(config),
    },
    {
        "name": "encoder type matches family",
        "when": "model.name in ['spatial', 'token', 'window_attention', 'operator']",
        "require": "model.<family>.encoder.type is supported by that family",
        "applies": lambda config: _model_family(config) in ENCODER_FAMILIES,
        "valid": lambda config: _valid_encoder_type(config),
        "message": lambda config: _encoder_type_message(config),
    },
    {
        "name": "shared MLP decoder config is valid",
        "when": "model.name in ['spatial', 'token', 'window_attention', 'operator']",
        "require": "decoder hidden/num_layers/neck are positive integers and dropout is in [0, 1)",
        "applies": lambda config: _model_family(config) in ENCODER_FAMILIES,
        "valid": lambda config: _valid_shared_decoder_cfg(config),
        "message": lambda config: _shared_decoder_cfg_message(config),
    },
    {
        "name": "spatial convolution parameters",
        "when": "model.name == 'spatial'",
        "require": "spatial residual is boolean; dilation and stride are positive integers",
        "applies": lambda config: _model_family(config) == "spatial",
        "valid": lambda config: _valid_conv_cfg(_encoder_cfg(config)),
        "message": lambda config: _conv_cfg_message(config),
    },
    {
        "name": "UNet GroupNorm channels",
        "when": "model.name == 'spatial' and model.spatial.encoder.type == 'UNet' and use_gn == true",
        "require": "all hidden * 2^i channel counts are divisible by min(8, channels)",
        "applies": lambda config: (
            _model_family(config) == "spatial"
            and _encoder_type(config) == "UNet"
            and _encoder_cfg(config).get("use_gn", False)
        ),
        "valid": lambda config: _valid_unet_groupnorm(config),
        "message": lambda config: _unet_groupnorm_message(config),
    },
    {
        "name": "Mamba basic parameters",
        "when": "model.name == 'token' and model.token.encoder.type == 'Mamba'",
        "require": "Mamba integer fields are positive; d_conv in [2, 4]; dropout in [0, 1); scan_mode/variant are supported",
        "applies": lambda config: _model_family(config) == "token" and _encoder_type(config) == "Mamba",
        "valid": lambda config: _valid_mamba_basic(config),
        "message": lambda config: _mamba_basic_message(config),
    },
    {
        "name": "sequence model basic parameters",
        "when": "model.name == 'token' and encoder.type in ['RNN', 'LSTM', 'Transformer']",
        "require": "integer fields are positive; dropout in [0, 1); scan_mode is supported",
        "applies": lambda config: _model_family(config) == "token" and _encoder_type(config) in ("RNN", "LSTM", "Transformer"),
        "valid": lambda config: _valid_sequence_basic(config),
        "message": lambda config: _sequence_basic_message(config),
    },
    {
        "name": "Transformer head dimension",
        "when": "model.name == 'token' and encoder.type == 'Transformer'",
        "require": "enc_ch % nhead == 0",
        "applies": lambda config: _model_family(config) == "token" and _encoder_type(config) == "Transformer",
        "valid": lambda config: _token_cfg(config).get("enc_ch", 64) % _token_cfg(config).get("nhead", 4) == 0,
        "message": lambda config: (
            "model.token.encoder requires enc_ch % nhead == 0, "
            f"got enc_ch={_token_cfg(config).get('enc_ch', 64)}, "
            f"nhead={_token_cfg(config).get('nhead', 4)}."
        ),
    },
    {
        "name": "window attention basic parameters",
        "when": "model.name == 'window_attention'",
        "require": "window-attention integer fields are positive; dropout in [0, 1); mlp_ratio is positive",
        "applies": lambda config: _model_family(config) == "window_attention",
        "valid": lambda config: _valid_window_attention_basic(config),
        "message": lambda config: _window_attention_basic_message(config),
    },
    {
        "name": "window attention head dimension",
        "when": "model.name == 'window_attention'",
        "require": "enc_ch % nhead == 0",
        "applies": lambda config: _model_family(config) == "window_attention",
        "valid": lambda config: _window_cfg(config).get("enc_ch", 64) % _window_cfg(config).get("nhead", 4) == 0,
        "message": lambda config: (
            "model.window_attention.encoder requires enc_ch % nhead == 0, "
            f"got enc_ch={_window_cfg(config).get('enc_ch', 64)}, "
            f"nhead={_window_cfg(config).get('nhead', 4)}."
        ),
    },
    {
        "name": "operator basic parameters",
        "when": "model.name == 'operator'",
        "require": "operator integer fields are positive and dropout is in [0, 1)",
        "applies": lambda config: _model_family(config) == "operator",
        "valid": lambda config: _valid_operator_basic(config),
        "message": lambda config: _operator_basic_message(config),
    },
    {
        "name": "Mamba2 head dimension",
        "when": "model.name == 'token' and encoder.type == 'Mamba' and variant == 'mamba2'",
        "require": "(expand * enc_ch) % headdim == 0",
        "applies": lambda config: (
            _model_family(config) == "token"
            and _encoder_type(config) == "Mamba"
            and _mamba_cfg(config).get("variant", "mamba") == "mamba2"
        ),
        "valid": lambda config: _mamba_d_inner(config) % _mamba_cfg(config).get("headdim", 16) == 0,
        "message": lambda config: (
            "model.token.encoder with variant='mamba2' requires (expand * enc_ch) % headdim == 0, "
            f"got enc_ch={_mamba_cfg(config).get('enc_ch', 64)}, "
            f"expand={_mamba_cfg(config).get('expand', 2)}, "
            f"headdim={_mamba_cfg(config).get('headdim', 16)}."
        ),
    },
    {
        "name": "Mamba2 accelerated projection width",
        "when": "model.name == 'token' and encoder.type == 'Mamba' and variant == 'mamba2'",
        "require": "2 * expand * enc_ch + 2 * d_state + nheads is divisible by 8",
        "applies": lambda config: (
            _model_family(config) == "token"
            and _encoder_type(config) == "Mamba"
            and _mamba_cfg(config).get("variant", "mamba") == "mamba2"
        ),
        "valid": lambda config: _mamba_in_proj_width(config) % 8 == 0,
        "message": lambda config: (
            "model.token.encoder with variant='mamba2' requires accelerated path alignment: "
            "2 * expand * enc_ch + 2 * d_state + nheads to be divisible by 8, "
            f"got {_mamba_in_proj_width(config)} (nheads={_mamba_nheads(config)})."
        ),
    },
    {
        "name": "voxelwise config is valid",
        "when": "model.name == 'voxelwise'",
        "require": "voxelwise MLP numeric fields are valid",
        "applies": lambda config: _model_family(config) == "voxelwise",
        "valid": lambda config: _valid_voxelwise_cfg(config),
        "message": lambda config: _voxelwise_cfg_message(config),
    },
    {
        "name": "CoordINR config shape is valid",
        "when": "model.name == 'coords'",
        "require": "model.coords input, encoder, and decoder are dictionaries",
        "applies": lambda config: _model_family(config) == "coords",
        "valid": lambda config: _valid_coord_cfg_shape(config),
        "message": lambda config: _coord_cfg_shape_message(config),
    },
    {
        "name": "CoordINR encoder mode is valid",
        "when": "model.name == 'coords'",
        "require": "model.coords.encoder.mode is a valid string or concat list",
        "applies": lambda config: _model_family(config) == "coords",
        "valid": lambda config: _valid_coord_encoder_mode(config),
        "message": lambda config: _coord_encoder_mode_message(config),
    },
    {
        "name": "CoordINR hash-grid feature width is tiny-cuda-nn compatible",
        "when": "model.name == 'coords' and any coord encoder mode is 'hash_grid'",
        "require": "hash_grid n_features_per_level is one of 1, 2, 4, or 8",
        "applies": lambda config: _model_family(config) == "coords",
        "valid": lambda config: _valid_coord_hash_grid_features(config),
        "message": lambda config: _coord_hash_grid_features_message(config),
    },
    {
        "name": "CoordINR decoder config is valid",
        "when": "model.name == 'coords'",
        "require": "model.coords.decoder.activation is supported and residual/use_KAN are boolean",
        "applies": lambda config: _model_family(config) == "coords",
        "valid": lambda config: _valid_coord_decoder_cfg(config),
        "message": lambda config: _coord_decoder_cfg_message(config),
    },
]


def validate_model_constraints(gen_config):
    model_cfg = gen_config.get("model", {})
    if not isinstance(model_cfg, dict):
        return False, "model must be a dictionary."

    for rule in MODEL_CONSTRAINTS:
        if rule["applies"](gen_config) and not rule["valid"](gen_config):
            return False, rule["message"](gen_config)
    return True, "Valid"


def _model_cfg(config):
    return config.get("model", {})


def _model_family(config):
    return _model_cfg(config).get("name")


def _dataset_type(config):
    return config.get("data", {}).get("dataset", "patchwise")


def _family_cfg(config, family=None):
    family = _model_family(config) if family is None else family
    cfg = _model_cfg(config).get(family, {})
    return cfg if isinstance(cfg, dict) else cfg


def _encoder_cfg(config):
    cfg = _family_cfg(config)
    if not isinstance(cfg, dict):
        return {}
    encoder = cfg.get("encoder", {})
    return encoder if isinstance(encoder, dict) else {}


def _decoder_cfg(config):
    cfg = _family_cfg(config)
    if not isinstance(cfg, dict):
        return {}
    decoder = cfg.get("decoder", {})
    return decoder if isinstance(decoder, dict) else {}


def _encoder_type(config):
    family = _model_family(config)
    defaults = {
        "spatial": "CNN",
        "token": "Mamba",
        "window_attention": "Swin",
        "operator": "FNO",
    }
    return _encoder_cfg(config).get("type", defaults.get(family))


def _allowed_encoder_types(family):
    return {
        "spatial": SPATIAL_ENCODERS,
        "token": TOKEN_ENCODERS,
        "window_attention": WINDOW_ENCODERS,
        "operator": OPERATOR_ENCODERS,
    }.get(family, ())


def _encoder_decoder_cfg_shape_error(config):
    family = _model_family(config)
    cfg = _family_cfg(config)
    if not isinstance(cfg, dict):
        return f"model.{family} must be a dictionary"
    for key in ("encoder", "decoder"):
        value = cfg.get(key, {})
        if value is not None and not isinstance(value, dict):
            return f"model.{family}.{key} must be a dictionary"
    for key in ("use_KAN", "use_VAE"):
        value = cfg.get(key, False)
        if not isinstance(value, bool):
            return f"model.{family}.{key} must be boolean"
    return None


def _valid_encoder_decoder_cfg_shape(config):
    return _encoder_decoder_cfg_shape_error(config) is None


def _encoder_decoder_cfg_shape_message(config):
    return f"Invalid model.{_model_family(config)} config: {_encoder_decoder_cfg_shape_error(config)}."


def _valid_encoder_type(config):
    family = _model_family(config)
    enc_type = _encoder_type(config)
    return enc_type in _allowed_encoder_types(family)


def _encoder_type_message(config):
    family = _model_family(config)
    return (
        f"model.{family}.encoder.type must be one of {_allowed_encoder_types(family)}, "
        f"got {_encoder_type(config)!r}."
    )


def _positive_int(value):
    return isinstance(value, int) and value > 0


def _valid_conv_cfg(cfg):
    if not isinstance(cfg, dict):
        return False
    enc_type = cfg.get("type", "CNN")
    fields = ("enc_ch", "enc_layers") if enc_type == "CNN" else ("hidden", "num_layers")
    for field in fields:
        if field in cfg and not _positive_int(cfg[field]):
            return False
    if not isinstance(cfg.get("residual", False), bool):
        return False
    for field in ("dilation", "stride"):
        if not _positive_int(cfg.get(field, 1)):
            return False
    return True


def _conv_cfg_message(config):
    cfg = _encoder_cfg(config)
    enc_type = _encoder_type(config)
    fields = ("enc_ch", "enc_layers") if enc_type == "CNN" else ("hidden", "num_layers")
    for field in fields:
        if field in cfg and not _positive_int(cfg[field]):
            return f"model.spatial.encoder.{field} must be a positive integer, got {cfg[field]!r}."
    residual = cfg.get("residual", False)
    if not isinstance(residual, bool):
        return f"model.spatial.encoder.residual must be boolean, got {residual!r}."
    for field in ("dilation", "stride"):
        value = cfg.get(field, 1)
        if not _positive_int(value):
            return f"model.spatial.encoder.{field} must be a positive integer, got {value!r}."
    return "Invalid model.spatial.encoder convolution configuration."


def _valid_shared_decoder_cfg(config):
    decoder = _decoder_cfg(config)
    if not isinstance(decoder, dict):
        return False
    for field, default in {"hidden": 256, "num_layers": 5, "neck": 64}.items():
        value = decoder.get(field, default)
        if not _positive_int(value):
            return False
    dropout = decoder.get("dropout", 0.0)
    return isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0


def _shared_decoder_cfg_message(config):
    family = _model_family(config)
    decoder = _decoder_cfg(config)
    if not isinstance(decoder, dict):
        return f"model.{family}.decoder must be a dictionary."
    for field, default in {"hidden": 256, "num_layers": 5, "neck": 64}.items():
        value = decoder.get(field, default)
        if not _positive_int(value):
            return f"model.{family}.decoder.{field} must be a positive integer, got {value!r}."
    dropout = decoder.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.{family}.decoder.dropout must be in [0, 1), got {dropout!r}."
    return f"Invalid model.{family}.decoder configuration."


def _valid_unet_groupnorm(config):
    cfg = _encoder_cfg(config)
    hidden = cfg.get("hidden", 64)
    num_layers = cfg.get("num_layers", 3)
    if not _positive_int(hidden):
        return False
    if not isinstance(num_layers, int) or num_layers < 0:
        return False
    channels = [hidden * (2 ** i) for i in range(num_layers + 1)]
    return all(c % min(8, c) == 0 for c in channels)


def _unet_groupnorm_message(config):
    cfg = _encoder_cfg(config)
    hidden = cfg.get("hidden", 64)
    num_layers = cfg.get("num_layers", 3)
    if not _positive_int(hidden):
        return f"model.spatial.encoder.hidden must be a positive integer, got {hidden!r}."
    if not isinstance(num_layers, int) or num_layers < 0:
        return f"model.spatial.encoder.num_layers must be a non-negative integer, got {num_layers!r}."
    channels = [hidden * (2 ** i) for i in range(num_layers + 1)]
    invalid = [(c, min(8, c)) for c in channels if c % min(8, c) != 0]
    bad = ", ".join(f"{c} channels with {g} groups" for c, g in invalid)
    return (
        "Invalid model.spatial GroupNorm configuration: use_gn=true requires every "
        f"UNet block channel count to be divisible by min(8, channels). Offending channel counts: {bad}."
    )


def _token_cfg(config):
    return _encoder_cfg(config)


def _mamba_cfg(config):
    return _encoder_cfg(config)


def _sequence_positive_int_fields(model_name):
    fields = {"enc_ch": 64, "depth": 2, "stem_layers": 1}
    if model_name == "Transformer":
        fields.update({"nhead": 4, "dim_feedforward": 256})
    return fields


def _valid_sequence_basic(config):
    name = _encoder_type(config)
    cfg = _token_cfg(config)
    for field, default in _sequence_positive_int_fields(name).items():
        if not _positive_int(cfg.get(field, default)):
            return False
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return False
    return cfg.get("scan_mode", "bidirectional") in ("flatten", "bidirectional", "axial")


def _sequence_basic_message(config):
    name = _encoder_type(config)
    cfg = _token_cfg(config)
    for field, default in _sequence_positive_int_fields(name).items():
        value = cfg.get(field, default)
        if not _positive_int(value):
            return f"model.token.encoder.{field} must be a positive integer, got {value!r}."
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.token.encoder.dropout must be in [0, 1), got {dropout!r}."
    scan_mode = cfg.get("scan_mode", "bidirectional")
    if scan_mode not in ("flatten", "bidirectional", "axial"):
        return (
            "model.token.encoder.scan_mode must be one of 'flatten', 'bidirectional', or 'axial', "
            f"got {scan_mode!r}."
        )
    return "Invalid model.token.encoder configuration."


def _mamba_positive_int_fields():
    return {
        "enc_ch": 64,
        "depth": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "stem_layers": 1,
        "headdim": 16,
    }


def _valid_mamba_basic(config):
    cfg = _mamba_cfg(config)
    for field, default in _mamba_positive_int_fields().items():
        if not _positive_int(cfg.get(field, default)):
            return False
    d_conv = cfg.get("d_conv", 4)
    if d_conv < 2 or d_conv > 4:
        return False
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return False
    if cfg.get("scan_mode", "bidirectional") not in ("flatten", "bidirectional", "axial"):
        return False
    return cfg.get("variant", "mamba") in ("mamba", "mamba2")


def _mamba_basic_message(config):
    cfg = _mamba_cfg(config)
    for field, default in _mamba_positive_int_fields().items():
        value = cfg.get(field, default)
        if not _positive_int(value):
            return f"model.token.encoder.{field} must be a positive integer, got {value!r}."
    d_conv = cfg.get("d_conv", 4)
    if d_conv < 2 or d_conv > 4:
        return f"model.token.encoder.d_conv must be 2, 3, or 4, got {d_conv!r}."
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.token.encoder.dropout must be in [0, 1), got {dropout!r}."
    scan_mode = cfg.get("scan_mode", "bidirectional")
    if scan_mode not in ("flatten", "bidirectional", "axial"):
        return (
            "model.token.encoder.scan_mode must be one of 'flatten', 'bidirectional', or 'axial', "
            f"got {scan_mode!r}."
        )
    variant = cfg.get("variant", "mamba")
    return f"model.token.encoder.variant must be one of 'mamba' or 'mamba2', got {variant!r}."


def _window_cfg(config):
    return _encoder_cfg(config)


def _window_positive_int_fields(name):
    fields = {"enc_ch": 64, "depth": 2, "nhead": 4, "stem_layers": 1}
    if name == "Swin":
        fields["window_size"] = 4
    elif name == "MaxViT":
        fields["block_size"] = 4
        fields["grid_size"] = 4
    elif name == "CSWin":
        fields["stripe_size"] = 4
    elif name == "Twins":
        fields["window_size"] = 4
        fields["sample_stride"] = 4
    elif name == "FocalTransformer":
        fields["window_size"] = 4
        fields["focal_stride"] = 2
    return fields


def _valid_window_attention_basic(config):
    name = _encoder_type(config)
    cfg = _window_cfg(config)
    for field, default in _window_positive_int_fields(name).items():
        if not _positive_int(cfg.get(field, default)):
            return False
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return False
    mlp_ratio = cfg.get("mlp_ratio", 4.0)
    return isinstance(mlp_ratio, (int, float)) and mlp_ratio > 0


def _window_attention_basic_message(config):
    name = _encoder_type(config)
    cfg = _window_cfg(config)
    for field, default in _window_positive_int_fields(name).items():
        value = cfg.get(field, default)
        if not _positive_int(value):
            return f"model.window_attention.encoder.{field} must be a positive integer, got {value!r}."
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.window_attention.encoder.dropout must be in [0, 1), got {dropout!r}."
    mlp_ratio = cfg.get("mlp_ratio", 4.0)
    if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
        return f"model.window_attention.encoder.mlp_ratio must be positive, got {mlp_ratio!r}."
    return f"Invalid model.window_attention.encoder configuration for {name}."


def _operator_cfg(config):
    return _encoder_cfg(config)


def _operator_positive_int_fields(name):
    fields = {"enc_ch": 64, "depth": 4, "stem_layers": 1}
    if name in ("FNO", "UFNO", "U-FNO", "UNO"):
        fields["modes"] = 12
    if name == "AFNO":
        fields["num_blocks"] = 8
    return fields


def _valid_operator_basic(config):
    name = _encoder_type(config)
    cfg = _operator_cfg(config)
    for field, default in _operator_positive_int_fields(name).items():
        if not _positive_int(cfg.get(field, default)):
            return False
    if name == "AFNO":
        enc_ch = cfg.get("enc_ch", 64)
        num_blocks = cfg.get("num_blocks", 8)
        if enc_ch % num_blocks != 0:
            return False
        for field in ("sparsity_threshold", "hard_thresholding_fraction"):
            value = cfg.get(field, 0.01 if field == "sparsity_threshold" else 1.0)
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                return False
    dropout = cfg.get("dropout", 0.0)
    return isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0


def _operator_basic_message(config):
    name = _encoder_type(config)
    cfg = _operator_cfg(config)
    for field, default in _operator_positive_int_fields(name).items():
        value = cfg.get(field, default)
        if not _positive_int(value):
            return f"model.operator.encoder.{field} must be a positive integer, got {value!r}."
    if name == "AFNO":
        enc_ch = cfg.get("enc_ch", 64)
        num_blocks = cfg.get("num_blocks", 8)
        if enc_ch % num_blocks != 0:
            return f"model.operator.encoder requires enc_ch % num_blocks == 0, got {enc_ch}, {num_blocks}."
        for field in ("sparsity_threshold", "hard_thresholding_fraction"):
            value = cfg.get(field, 0.01 if field == "sparsity_threshold" else 1.0)
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                return f"model.operator.encoder.{field} must be in [0, 1], got {value!r}."
    dropout = cfg.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.operator.encoder.dropout must be in [0, 1), got {dropout!r}."
    return f"Invalid model.operator.encoder configuration for {name}."


def _valid_voxelwise_cfg(config):
    cfg = _family_cfg(config)
    for field, default in {"hidden": 128, "num_layers": 3, "mlp_neck": 64}.items():
        if not _positive_int(cfg.get(field, default)):
            return False
    dropout = cfg.get("mlp_dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return False
    return isinstance(cfg.get("use_KAN", False), bool)


def _voxelwise_cfg_message(config):
    cfg = _family_cfg(config)
    for field, default in {"hidden": 128, "num_layers": 3, "mlp_neck": 64}.items():
        value = cfg.get(field, default)
        if not _positive_int(value):
            return f"model.voxelwise.{field} must be a positive integer, got {value!r}."
    dropout = cfg.get("mlp_dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"model.voxelwise.mlp_dropout must be in [0, 1), got {dropout!r}."
    use_kan = cfg.get("use_KAN", False)
    if not isinstance(use_kan, bool):
        return f"model.voxelwise.use_KAN must be boolean, got {use_kan!r}."
    return "Invalid model.voxelwise configuration."


def _coord_cfg(config):
    cfg = _family_cfg(config, "coords")
    return cfg if isinstance(cfg, dict) else {}


def _coord_cfg_shape_error(config):
    cfg = _coord_cfg(config)
    for key in ("input", "encoder", "decoder"):
        value = cfg.get(key, {})
        if value is not None and not isinstance(value, dict):
            return f"model.coords.{key} must be a dictionary"
    use_vae = cfg.get("use_VAE", False)
    if not isinstance(use_vae, bool):
        return f"model.coords.use_VAE must be boolean, got {use_vae!r}"
    return None


def _valid_coord_cfg_shape(config):
    return _coord_cfg_shape_error(config) is None


def _coord_cfg_shape_message(config):
    return f"Invalid model.coords config: {_coord_cfg_shape_error(config)}."


def _coord_encoder_cfg(config):
    encoder = _coord_cfg(config).get("encoder", {})
    return encoder if isinstance(encoder, dict) else {}


def _coord_decoder_cfg(config):
    decoder = _coord_cfg(config).get("decoder", {})
    return decoder if isinstance(decoder, dict) else {}


def _coord_encoder_base_modes():
    return {
        "none",
        "positional_encoding",
        "gaussian_fourier",
        "siren_first_layer",
        "xy_yz_xz",
        "xyz_plus_xy_yz_xz",
        "dense_grid",
        "multi_res_grid",
        "hash_grid",
        "tri_plane",
    }


def _coord_encoder_mode_aliases():
    return {
        "fourier": "gaussian_fourier",
        "triplane": "tri_plane",
        "hash": "hash_grid",
        "grid": "dense_grid",
        "multires_grid": "multi_res_grid",
    }


def _canonical_coord_encoder_mode(mode):
    return _coord_encoder_mode_aliases().get(mode, mode)


def _coord_encoder_mode_error(config):
    mode = _coord_encoder_cfg(config).get("mode", "hash_grid")
    allowed = _coord_encoder_base_modes()

    if isinstance(mode, str):
        mode = _canonical_coord_encoder_mode(mode)
        if mode not in allowed:
            return f"unknown encoder mode {mode!r}"
        return None

    if not isinstance(mode, list):
        return "encoder.mode must be a string or a list"
    if len(mode) < 2:
        return "encoder.mode list must contain at least two encoders"

    names = []
    for item in mode:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("mode")
        else:
            return f"encoder.mode list entries must be strings or dictionaries, got {type(item).__name__}"
        if name is None:
            return f"encoder spec is missing 'mode': {item!r}"
        name = _canonical_coord_encoder_mode(name)
        if name not in allowed:
            return f"unknown encoder mode {name!r}"
        names.append(name)

    if "none" in names:
        return "'none' cannot be used inside encoder.mode concat lists"
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return f"duplicate encoder modes are not allowed in concat lists: {duplicates}"
    coord_pair_modes = {"xy_yz_xz", "xyz_plus_xy_yz_xz"}
    if len(coord_pair_modes.intersection(names)) > 1:
        return "choose only one of 'xy_yz_xz' or 'xyz_plus_xy_yz_xz' in a concat list"
    return None


def _valid_coord_encoder_mode(config):
    return _coord_encoder_mode_error(config) is None


def _coord_encoder_mode_message(config):
    return f"Invalid model.coords.encoder.mode: {_coord_encoder_mode_error(config)}."


def _iter_coord_encoder_specs(config):
    encoder = _coord_encoder_cfg(config)
    mode = encoder.get("mode", "hash_grid")

    if isinstance(mode, str):
        yield "model.coords.encoder", _canonical_coord_encoder_mode(mode), encoder
        return

    if not isinstance(mode, list):
        return

    for idx, item in enumerate(mode):
        path = f"model.coords.encoder.mode[{idx}]"
        if isinstance(item, str):
            yield path, _canonical_coord_encoder_mode(item), {}
        elif isinstance(item, dict):
            item_mode = item.get("mode")
            if item_mode is not None:
                yield path, _canonical_coord_encoder_mode(item_mode), item


def _coord_hash_grid_features_error(config):
    if _coord_encoder_mode_error(config) is not None:
        return None

    for path, mode, cfg in _iter_coord_encoder_specs(config):
        if mode != "hash_grid":
            continue
        value = cfg.get("n_features_per_level", 2)
        if type(value) is not int or value not in HASH_GRID_N_FEATURES_PER_LEVEL:
            allowed = ", ".join(str(v) for v in HASH_GRID_N_FEATURES_PER_LEVEL)
            return f"{path}.n_features_per_level must be one of {allowed}, got {value!r}"
    return None


def _valid_coord_hash_grid_features(config):
    return _coord_hash_grid_features_error(config) is None


def _coord_hash_grid_features_message(config):
    return f"Invalid model.coords.encoder hash_grid config: {_coord_hash_grid_features_error(config)}."


def _coord_decoder_cfg_error(config):
    decoder = _coord_decoder_cfg(config)
    activation = decoder.get("activation", "relu")
    if activation not in {"relu", "softplus", "silu", "gelu", "sine", "leakyrelu"}:
        return f"unsupported activation {activation!r}"
    residual = decoder.get("residual", False)
    if not isinstance(residual, bool):
        return f"residual must be boolean, got {residual!r}"
    use_kan = decoder.get("use_KAN", False)
    if not isinstance(use_kan, bool):
        return f"use_KAN must be boolean, got {use_kan!r}"
    if use_kan and residual:
        return "decoder.use_KAN=true is not compatible with decoder.residual=true"
    for field, default in {"hidden": 256, "num_layers": 4}.items():
        value = decoder.get(field, default)
        if not _positive_int(value):
            return f"{field} must be a positive integer, got {value!r}"
    dropout = decoder.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout >= 1.0:
        return f"dropout must be in [0, 1), got {dropout!r}"
    if "mode" in decoder:
        return "decoder.mode is deprecated; use decoder.activation and decoder.residual"
    return None


def _valid_coord_decoder_cfg(config):
    return _coord_decoder_cfg_error(config) is None


def _coord_decoder_cfg_message(config):
    return f"Invalid model.coords.decoder: {_coord_decoder_cfg_error(config)}."


def _mamba_d_inner(config):
    cfg = _mamba_cfg(config)
    return cfg.get("expand", 2) * cfg.get("enc_ch", 64)


def _mamba_nheads(config):
    return _mamba_d_inner(config) // _mamba_cfg(config).get("headdim", 16)


def _mamba_in_proj_width(config):
    cfg = _mamba_cfg(config)
    return 2 * _mamba_d_inner(config) + 2 * cfg.get("d_state", 16) + _mamba_nheads(config)
