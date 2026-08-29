import numpy as np
import torch

from library.dataio.diffusion import coords, patchwise, slicewise, voxelwise
from library.utility.qtlib import block2brain


def _uses_coords(model) -> bool:
    return model.__class__.__name__ == "CoordINR"


def _batch_model_input(model, batch: dict):
    key = "coords" if _uses_coords(model) else "signal"
    if key not in batch:
        raise KeyError(
            f"Model '{model.__class__.__name__}' expects batch['{key}'], "
            f"but batch only has keys: {list(batch.keys())}"
        )
    return batch[key]


def infer_volume(cfg, model, volume, out_dir):
    
    physics_name   = cfg["physics"]["name"]
    physics_module = __import__(f"library.physics.{physics_name}", fromlist=[physics_name])
    physics_class  = getattr(physics_module, physics_name)

    batch_size = cfg["data"].get("batch_size", 4)

    if isinstance(volume, slicewise):
        output = _infer_slicewise(model, volume, batch_size, physics_class.N_PARAMS)
    elif isinstance(volume, voxelwise):
        output = _infer_voxelwise(model, volume, batch_size)
    elif isinstance(volume, coords):
        output = _infer_coords(model, volume, batch_size)
    elif isinstance(volume, patchwise):
        output = _infer_patchwise(model, volume, batch_size)
    else:
        raise TypeError(f"Unsupported volume type: {type(volume)}")
    
    require_S0 = getattr(physics_class, "require_S0", False)
    
    if require_S0:
        signal = volume._volume.cpu()
        signal_flat = signal[volume._mask.cpu().bool()]
        S0_flat = signal_flat[:, volume.bval==0].mean(dim=1)
        params_flat = physics_class.output_to_param(torch.tensor(output), S0=S0_flat).numpy()
    else:
        params_flat = physics_class.output_to_param(torch.tensor(output)).numpy()

    physics_class.save_nifti(
        params_flat,
        volume.orig_mask,       # property：从 _bbox + _mask 重建，不额外存储
        volume.affine,
        out_dir,
        volume.norm_factor,
    )


def _infer_slicewise(model, volume, batch_size: int, n_params: int) -> np.ndarray:
    axis = volume._axis
    vol_shape = volume._volume.shape[:3]
    all_ids = sorted(volume._all_ids)
    params_crop = torch.zeros(*vol_shape, n_params, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_ids), batch_size):
            ids = all_ids[start:start + batch_size]
            out = model(volume.sample_batch(ids)["signal"])
            for j, i in enumerate(ids):
                p = out[j].permute(1, 2, 0).cpu()
                if axis == 0:
                    params_crop[i] = p
                elif axis == 1:
                    params_crop[:, i] = p
                else:
                    params_crop[:, :, i] = p

    mask_crop = volume._mask.cpu().bool().numpy()
    return params_crop.numpy()[mask_crop]


def _infer_patchwise(model, volume, batch_size: int) -> np.ndarray:
    params_vol_all = []
    all_ids = list(range(len(volume.ind_block)))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_ids), batch_size):
            ids = all_ids[start:start + batch_size]
            batch = volume.sample_batch(ids)
            out = model(batch["signal"])
            params_vol_all.append(out.permute(0, 2, 3, 4, 1).cpu().numpy())

    params_vol = np.concatenate(params_vol_all, axis=0)
    mask_np = volume._mask_crop_np[..., np.newaxis]
    vol_brain, _ = block2brain(params_vol, volume.ind_block, mask_np)
    return vol_brain[volume._mask_crop_np]


def _infer_voxelwise(model, volume, batch_size: int) -> np.ndarray:
    n = volume._brain.shape[0]
    results = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ids = list(range(start, min(start + batch_size, n)))
            out = model(volume.sample_batch(ids)["signal"])
            results.append(out.cpu())

    return torch.cat(results, dim=0).numpy()


def _infer_coords(model, volume, batch_size: int) -> np.ndarray:
    n = volume._brain.shape[0]
    results = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ids = list(range(start, min(start + batch_size, n)))
            batch = volume.sample_batch(ids)
            out = model(_batch_model_input(model, batch))
            results.append(out.cpu())

    return torch.cat(results, dim=0).numpy()
