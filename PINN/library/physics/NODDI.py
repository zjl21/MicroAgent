import os
import math
from glob import glob

import nibabel as nib
import numpy as np
import scipy.special
import torch


# ————————————————————————
# ————————————————————————

def _legendre_gaussian_integral(x, n=6):
    """
    Integral terms used by the Watson spherical-harmonic stick signal.

    x: (N, B), where N is the number of diffusion measurements and B is the
       number of voxels.
    return: (N, n + 1, B), corresponding to even Legendre orders 0..12.
    """
    if n > 6:
        raise ValueError("NODDI implementation supports Legendre terms only up to order 12")

    eps = torch.finfo(x.dtype).eps
    x_safe = x.clamp_min(eps)
    sqrtx = torch.sqrt(x_safe)

    # Exact recurrence is stable away from zero.  For x near zero, use the
    # Taylor expansion from the MATLAB/NODDI implementation to avoid 0/0 terms.
    I = []
    I0 = math.sqrt(math.pi) * torch.erf(sqrtx) / sqrtx
    I.append(I0)
    emx = -torch.exp(-x_safe)
    inv_x = 1.0 / x_safe
    for i in range(1, n + 1):
        I.append((emx + (i - 0.5) * I[i - 1]) * inv_x)

    L_exact = torch.stack(
        [
            I[0],
            -0.5 * I[0] + 1.5 * I[1],
            0.375 * I[0] - 3.75 * I[1] + 4.375 * I[2],
            -0.3125 * I[0] + 6.5625 * I[1] - 19.6875 * I[2] + 14.4375 * I[3],
            0.2734375 * I[0]
            - 9.84375 * I[1]
            + 54.140625 * I[2]
            - 93.84375 * I[3]
            + 50.2734375 * I[4],
            -(63 / 256) * I[0]
            + (3465 / 256) * I[1]
            - (30030 / 256) * I[2]
            + (90090 / 256) * I[3]
            - (109395 / 256) * I[4]
            + (46189 / 256) * I[5],
            (231 / 1024) * I[0]
            - (18018 / 1024) * I[1]
            + (225225 / 1024) * I[2]
            - (1021020 / 1024) * I[3]
            + (2078505 / 1024) * I[4]
            - (1939938 / 1024) * I[5]
            + (676039 / 1024) * I[6],
        ],
        dim=1,
    )

    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    L_approx = torch.stack(
        [
            2 - 2 * x / 3 + x2 / 5 - x3 / 21 + x4 / 108,
            -4 * x / 15 + 4 * x2 / 35 - 2 * x3 / 63 + 2 * x4 / 297,
            8 * x2 / 315 - 8 * x3 / 693 + 4 * x4 / 1287,
            -16 * x3 / 9009 + 16 * x4 / 19305,
            32 * x4 / 328185,
            -64 * x5 / 14549535,
            128 * x6 / 760543875,
        ],
        dim=1,
    )

    return torch.where((x > 0.05).unsqueeze(1), L_exact, L_approx)


def _legendre_even(cos_theta):
    """
    Even Legendre polynomials P_0, P_2, ..., P_12 evaluated at cos(theta).

    return: (N, 7, B)
    """
    return torch.stack(
        [
            torch.special.legendre_polynomial_p(cos_theta, 0),
            torch.special.legendre_polynomial_p(cos_theta, 2),
            torch.special.legendre_polynomial_p(cos_theta, 4),
            torch.special.legendre_polynomial_p(cos_theta, 6),
            torch.special.legendre_polynomial_p(cos_theta, 8),
            torch.special.legendre_polynomial_p(cos_theta, 10),
            torch.special.legendre_polynomial_p(cos_theta, 12),
        ],
        dim=1,
    )


def _erfi_interp(x, erfi_x, erfi_values, erfi_step):
    """Linear interpolation of erfi(x) on a precomputed lookup table."""
    table_x = erfi_x.to(device=x.device, dtype=x.dtype)
    table_y = erfi_values.to(device=x.device, dtype=x.dtype)
    step = x.new_tensor(erfi_step)

    x_clamped = x.clamp(table_x[0], table_x[-2])
    indices = torch.floor((x_clamped - table_x[0]) / step).long()
    weights = (x_clamped - table_x[indices]) / step
    return torch.lerp(table_y[indices], table_y[indices + 1], weights)


def _dawson_sqrt_kappa(kappa, erfi_x, erfi_values, erfi_step):
    """
    Dawson(sqrt(kappa)) used by Watson diffusivity expressions.

    For kappa <= 30, compute it from the erfi lookup table:
        Dawson(x) = 0.5 * sqrt(pi) * exp(-x^2) * erfi(x).
    For larger kappa, use the standard large-x asymptotic expansion to avoid
    overflow and to match the large-kappa handling in WatsonSHCoeff().
    """
    sqrt_k = torch.sqrt(kappa.clamp_min(torch.finfo(kappa.dtype).eps))
    erfi = _erfi_interp(sqrt_k, erfi_x, erfi_values, erfi_step)
    dawson_table = 0.5 * math.sqrt(math.pi) * torch.exp(-kappa) * erfi

    inv_x = 1.0 / sqrt_k
    inv_x2 = inv_x * inv_x
    dawson_asym = 0.5 * inv_x
    dawson_asym = dawson_asym + 0.25 * inv_x * inv_x2
    dawson_asym = dawson_asym + 0.375 * inv_x * inv_x2.pow(2)
    dawson_asym = dawson_asym + 0.9375 * inv_x * inv_x2.pow(3)
    dawson_asym = dawson_asym + 3.28125 * inv_x * inv_x2.pow(4)
    return torch.where(kappa <= 30.0, dawson_table, dawson_asym)


def _watson_hindered_diffusion_coeff(d_par, d_perp, kappa, erfi_x, erfi_values, erfi_step):
    """
    Apparent axial/radial diffusivities of the Watson-distributed hindered
    compartment.

    This is the device-independent version of WatsonHinderedDiffusionCoeff()
    in matlabnoddiparallelfast.py.
    """
    d_delta = d_par - d_perp
    d_trace = d_par + 2.0 * d_perp
    k2 = kappa * kappa

    dw_par_small = d_trace / 3.0 + 4.0 * d_delta * kappa / 45.0 + 8.0 * d_delta * k2 / 945.0
    dw_perp_small = d_trace / 3.0 - 2.0 * d_delta * kappa / 45.0 - 4.0 * d_delta * k2 / 945.0

    dawson = _dawson_sqrt_kappa(kappa, erfi_x, erfi_values, erfi_step)
    factor = torch.sqrt(kappa.clamp_min(torch.finfo(kappa.dtype).eps)) / dawson.clamp_min(
        torch.finfo(kappa.dtype).eps
    )
    dw_par_large = (-d_delta + 2.0 * d_perp * kappa + d_delta * factor) / (2.0 * kappa)
    dw_perp_large = (
        d_delta + 2.0 * (d_par + d_perp) * kappa - d_delta * factor
    ) / (4.0 * kappa)

    small = kappa < 1e-5
    return torch.where(small, dw_par_small, dw_par_large), torch.where(small, dw_perp_small, dw_perp_large)


def _watson_sh_coeff(kappa, erfi_x, erfi_values, erfi_step):
    """
    Watson distribution spherical-harmonic coefficients up to order 12.

    return: (B, 7), columns correspond to orders 0, 2, ..., 12.  The formula
    follows WatsonSHCoeff() from matlabnoddiparallelfast.py, with the exact
    branch limited to 0.1 < kappa <= 30 and the source polynomial branch used
    for kappa > 30.
    """
    k = kappa.reshape(-1)
    device, dtype = k.device, k.dtype
    C = torch.zeros((k.numel(), 7), device=device, dtype=dtype)
    C[:, 0] = 2.0 * math.sqrt(math.pi)

    k2 = k.pow(2)
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k
    k6 = k5 * k

    approx = k <= 0.1
    exact = (k > 0.1) & (k <= 30.0)
    large = k > 30.0

    if exact.any():
        ke = k[exact]
        sk = torch.sqrt(ke)
        sk2 = sk * ke
        sk3 = sk2 * ke
        sk4 = sk3 * ke
        sk5 = sk4 * ke
        sk6 = sk5 * ke
        erfi = _erfi_interp(sk, erfi_x, erfi_values, erfi_step)
        inv_erfi = 1.0 / erfi
        exp_k = torch.exp(ke)
        dawson = 0.5 * math.sqrt(math.pi) * erfi / exp_k

        C[exact, 1] = torch.sqrt(k.new_tensor(5.0)) * (
            3.0 * sk - (3.0 + 2.0 * ke) * dawson
        ) * exp_k * inv_erfi / ke

        C[exact, 2] = (
            (105.0 + 60.0 * ke + 12.0 * ke.pow(2)) * dawson
            - 105.0 * sk
            + 10.0 * sk2
        ) * 0.375 * exp_k * inv_erfi / ke.pow(2)

        C[exact, 3] = (
            (-3465.0 - 1890.0 * ke - 420.0 * ke.pow(2) - 40.0 * ke.pow(3)) * dawson
            + 3465.0 * sk
            - 420.0 * sk2
            + 84.0 * sk3
        ) * torch.sqrt(k.new_tensor(13.0 * math.pi)) / 64.0 / ke.pow(3) / dawson

        C[exact, 4] = torch.sqrt(k.new_tensor(17.0)) * (
            (675675.0 + 360360.0 * ke + 83160.0 * ke.pow(2) + 10080.0 * ke.pow(3) + 560.0 * ke.pow(4))
            * dawson
            - 675675.0 * sk
            + 90090.0 * sk2
            - 23100.0 * sk3
            + 744.0 * sk4
        ) * exp_k * inv_erfi / 512.0 / ke.pow(4)

        C[exact, 5] = (
            (-43648605.0 - 22972950.0 * ke - 5405400.0 * ke.pow(2) - 720720.0 * ke.pow(3) - 55440.0 * ke.pow(4) - 2016.0 * ke.pow(5))
            * dawson
            + 43648605.0 * sk
            - 6126120.0 * sk2
            + 1729728.0 * sk3
            - 82368.0 * sk4
            + 5104.0 * sk5
        ) * torch.sqrt(k.new_tensor(21.0 * math.pi)) / 4096.0 / ke.pow(5) / dawson

        C[exact, 6] = 5.0 * (
            (7027425405.0 + 3666482820.0 * ke + 872972100.0 * ke.pow(2) + 122522400.0 * ke.pow(3) + 10810800.0 * ke.pow(4) + 576576.0 * ke.pow(5) + 14784.0 * ke.pow(6))
            * dawson
            - 7027425405.0 * sk
            + 1018467450.0 * sk2
            - 302630328.0 * sk3
            + 17153136.0 * sk4
            - 1553552.0 * sk5
            + 25376.0 * sk6
        ) * exp_k * inv_erfi / 16384.0 / ke.pow(6)

    if large.any():
        lnkd = torch.log(k[large]) - math.log(30.0)
        lnkd2 = lnkd * lnkd
        lnkd3 = lnkd2 * lnkd
        lnkd4 = lnkd3 * lnkd
        lnkd5 = lnkd4 * lnkd
        lnkd6 = lnkd5 * lnkd

        C[large, 1] = 7.52308 + 0.411538 * lnkd - 0.214588 * lnkd2 + 0.0784091 * lnkd3 - 0.023981 * lnkd4 + 0.00731537 * lnkd5 - 0.0026467 * lnkd6
        C[large, 2] = 8.93718 + 1.62147 * lnkd - 0.733421 * lnkd2 + 0.191568 * lnkd3 - 0.0202906 * lnkd4 - 0.00779095 * lnkd5 + 0.00574847 * lnkd6
        C[large, 3] = 8.87905 + 3.35689 * lnkd - 1.15935 * lnkd2 + 0.0673053 * lnkd3 + 0.121857 * lnkd4 - 0.066642 * lnkd5 + 0.0180215 * lnkd6
        C[large, 4] = 7.84352 + 5.03178 * lnkd - 1.0193 * lnkd2 - 0.426362 * lnkd3 + 0.328816 * lnkd4 - 0.0688176 * lnkd5 - 0.0229398 * lnkd6
        C[large, 5] = 6.30113 + 6.09914 * lnkd - 0.16088 * lnkd2 - 1.05578 * lnkd3 + 0.338069 * lnkd4 + 0.0937157 * lnkd5 - 0.106935 * lnkd6
        C[large, 6] = 4.65678 + 6.30069 * lnkd + 1.13754 * lnkd2 - 1.38393 * lnkd3 - 0.0134758 * lnkd4 + 0.331686 * lnkd5 - 0.105954 * lnkd6

    if approx.any():
        ka = k[approx]
        C[approx, 1] = (4.0 / 3.0 * ka + 8.0 / 63.0 * k2[approx]) * math.sqrt(math.pi / 5.0)
        C[approx, 2] = (8.0 / 21.0 * k2[approx] + 32.0 / 693.0 * k3[approx]) * (math.sqrt(math.pi) * 0.2)
        C[approx, 3] = (16.0 / 693.0 * k3[approx] + 32.0 / 10395.0 * k4[approx]) * math.sqrt(math.pi / 13.0)
        C[approx, 4] = 32.0 / 19305.0 * k4[approx] * math.sqrt(math.pi / 17.0)
        C[approx, 5] = 64.0 * math.sqrt(math.pi / 21.0) * k5[approx] / 692835.0
        C[approx, 6] = 128.0 * math.sqrt(math.pi) * k6[approx] / 152108775.0

    return C


def sigmoid_relu_sigmoid_concat(x, upper=30):
    """
    Match the DIMOND/NODDI raw-to-physical kappa mapping used in the test code.

    x < 0      -> sigmoid(x)
    0 <= x < (upper-1) -> relu(x) + 0.5
    x >= (upper-1)     -> (upper-1) + sigmoid(x - (upper-1))
    """
    return torch.sigmoid(x) * (x < 0) + (torch.relu(x) + 0.5) * ((x > 0) & (x < (upper-1))) + ((upper-1) + torch.sigmoid(x - (upper-1))) * (x >= (upper-1))


def kappa_to_odi(kappa):
    """Convert Watson concentration kappa to ODI."""
    kappa = np.asarray(kappa)
    return (2.0 / np.pi) * np.arctan(1.0 / np.clip(kappa, 1e-12, None))


def dir_to_angles(fibdir):
    """Convert a unit direction vector to hemisphere-restricted spherical angles."""
    fibdir = np.asarray(fibdir)
    fibdir = fibdir / np.clip(np.linalg.norm(fibdir, axis=-1, keepdims=True), 1e-12, None)
    flip = fibdir[..., 2] < 0
    fibdir = fibdir.copy()
    fibdir[flip] *= -1.0
    theta = np.arccos(np.clip(fibdir[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(fibdir[..., 1], fibdir[..., 0]), 2.0 * np.pi)
    return theta, phi


def angles_to_dir(theta, phi):
    """Convert hemisphere-restricted spherical angles to a unit direction vector."""
    return np.stack(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ],
        axis=-1,
    )

# ————————————————————————
# physics
# ————————————————————————

class NODDI:
    """
    Watson-distributed NODDI forward model.

    Network-output parameter order:
        [fiso, ficvf, kappa, theta, phi]

    S0      : non-diffusion signal amplitude estimated from measured b0 signals.
    fiso    : isotropic/free-water volume fraction.
    ficvf   : intra-cellular volume fraction inside the non-isotropic tissue pool.
    kappa   : Watson concentration parameter.  The saved kappa maps are expected
              to already be in the physical scale used by the forward model.
    theta/phi: spherical angles of the principal fiber direction.  The
               network predicts angles because they are easier to constrain than
               an unconstrained 3-vector.

    Forward-model parameter order:
        [S0, fiso, ficvf, kappa, fibdir_x, fibdir_y, fibdir_z]

    output_to_param() is the boundary between these two conventions: it converts
    raw network theta/phi into a unit fiber direction and prepends the
    externally supplied b0-derived S0.
    """

    require_S0 = True
    PARAM_NAMES = ["fiso", "ficvf", "kappa", "theta", "phi"]
    N_PARAMS = len(PARAM_NAMES)

    @classmethod
    def recon_signal(cls, dpOut: str, dpDiff: str):
        """
        Reconstruct diffusion signal from saved NODDI parameter maps.
        """
        fpBval = glob(os.path.join(dpDiff, "*_diff.bval"))[0]
        fpBvec = glob(os.path.join(dpDiff, "*_diff.bvec"))[0]
        fpDiff = glob(os.path.join(dpDiff, "*_diff.nii.gz"))[0]

        bval = np.loadtxt(fpBval)
        bvec = np.loadtxt(fpBvec)
        if bvec.shape[0] != 3:
            bvec = bvec.T

        ref = nib.load(fpDiff)
        model = cls(bval, bvec)

        S0 = nib.load(os.path.join(dpOut, "S0.nii.gz")).get_fdata()
        fiso = nib.load(os.path.join(dpOut, "fiso.nii.gz")).get_fdata()
        ficvf = nib.load(os.path.join(dpOut, "ficvf.nii.gz")).get_fdata()
        kappa = nib.load(os.path.join(dpOut, "kappa.nii.gz")).get_fdata()
        fp_fiberdir = os.path.join(dpOut, "fiberdir.nii.gz")
        fibdir = nib.load(fp_fiberdir).get_fdata()

        params = np.concatenate(
            [
                S0[..., np.newaxis],
                fiso[..., np.newaxis],
                ficvf[..., np.newaxis],
                kappa[..., np.newaxis],
                fibdir,
            ],
            axis=3,
        )
        params = torch.from_numpy(params.reshape(-1, 7)).float()
        diff = model.forward(params).cpu().numpy().reshape(*S0.shape, -1)

        diff_img = nib.Nifti1Image(diff.astype(np.float32), ref.affine, ref.header)
        fpDiffPred = os.path.join(dpOut, "diff_pred.nii.gz")
        nib.save(diff_img, fpDiffPred)
        print(f"✅ 已保存重建的扩散信号到 {fpDiffPred}")

    @classmethod
    def output_to_param(
        cls,
        output: torch.Tensor,
        S0: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert raw network output to physical NODDI parameters.
        """

        # Fractions are bounded to [0, 1].
        fiso = torch.sigmoid(output[..., 0])
        ficvf = torch.sigmoid(output[..., 1])

        kappa = sigmoid_relu_sigmoid_concat(output[..., 2], upper=3.0) * 10
        # kappa = 30.0 * torch.sigmoid(output[..., 2])

        # theta is the polar angle relative to +z.  Restricting it to
        # [0, pi] makes the direction live on one full sphere coordinate
        # chart; phi covers azimuth in [0, 2*pi].
        theta = math.pi * torch.sigmoid(output[..., 3])
        phi = 2.0 * math.pi * torch.sigmoid(output[..., 4])

        fibdir = torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                S0.unsqueeze(-1),
                fiso.unsqueeze(-1),
                ficvf.unsqueeze(-1),
                kappa.unsqueeze(-1),
                fibdir,
            ],
            dim=-1,
        )

    @classmethod
    def save_nifti(
        cls,
        params_all: np.ndarray,
        mask: np.ndarray,
        affine: np.ndarray,
        out_dir: str,
        norm_factor=1.0,
    ):
        """
        Save NODDI physical parameters back to 3D/4D NIfTI volumes.
        """
        if hasattr(norm_factor, "item"):
            norm_factor = norm_factor.item()

        os.makedirs(out_dir, exist_ok=True)
        H, W, Z = mask.shape
        flat_mask = mask.reshape(-1)

        params_out = params_all.copy()
        params_out[:, 0] *= norm_factor

        scalar_names = ["S0", "fiso", "ficvf", "kappa"]
        for i, name in enumerate(scalar_names):
            arr = np.zeros(H * W * Z, dtype=np.float32)
            arr[flat_mask] = params_out[:, i]
            nib.save(nib.Nifti1Image(arr.reshape(H, W, Z), affine), os.path.join(out_dir, f"{name}.nii.gz"))

        fibdir = np.zeros((H * W * Z, 3), dtype=np.float32)
        fibdir_masked = params_out[:, 4:7]
        fibdir_masked = fibdir_masked / np.clip(
            np.linalg.norm(fibdir_masked, axis=1, keepdims=True), 1e-12, None
        )
        fibdir[flat_mask] = fibdir_masked
        fibdir_img = nib.Nifti1Image(fibdir.reshape(H, W, Z, 3), affine)
        nib.save(fibdir_img, os.path.join(out_dir, "fiberdir.nii.gz"))

        odi = np.zeros(H * W * Z, dtype=np.float32)
        odi[flat_mask] = kappa_to_odi(params_out[:, 3])
        nib.save(nib.Nifti1Image(odi.reshape(H, W, Z), affine), os.path.join(out_dir, "odi.nii.gz"))

    def __init__(
        self,
        bval,
        bvec,
        norm_factor=1.0,
        di=1.7,
        diso=3.0,
        erfi_grid_size=10000,
    ):
        """
        Build the PGSE protocol needed by the NODDI signal equations.

        bval: (N,)
            Diffusion weighting in s/mm^2, matching FSL .bval convention.
        bvec: (3, N) or (N, 3)
            Unit gradient directions.  b0 directions may be zero; they are
            replaced by [1, 0, 0] so direction normalization stays finite.
        norm_factor:
            Kept for consistency with DTI/DKI.  S0 is assumed to already be in
            the same normalized signal domain as the training data.
        di, diso:
            Fixed intrinsic and isotropic diffusivities in um^2/ms.
        erfi_grid_size:
            Number of samples in the erfi lookup table for 0 <= sqrt(kappa) <=
            sqrt(30).  kappa > 30 uses the asymptotic Watson coefficients from
            the source implementation and does not require erfi.
        """
        bval = torch.as_tensor(bval, dtype=torch.float32)
        bvec = torch.as_tensor(bvec, dtype=torch.float32)
        if bvec.shape[0] != 3:
            bvec = bvec.T

        self.norm_factor = norm_factor
        self.di = float(di)
        self.diso = float(diso)
        grad_dirs = bvec.T.contiguous()
        grad_dirs[bval == 0] = grad_dirs.new_tensor([1.0, 0.0, 0.0])
        grad_norm = torch.linalg.norm(grad_dirs, dim=1, keepdim=True).clamp_min(1e-12)
        grad_dirs = grad_dirs / grad_norm

        # build_blocks.py 会把 physics_instance.gradient 放到目标设备上。
        self.gradient = grad_dirs
        self.bval = bval / 1000.0  # convert to s/um^2 for um^2/ms diffusivities

        # imaginary error function, suitable for /kappa < 30
        erfi_x_np = np.linspace(0.0, math.sqrt(30.0), erfi_grid_size, dtype=np.float64)
        erfi_y_np = scipy.special.erfi(erfi_x_np).astype(np.float64)
        self.erfi_x = torch.from_numpy(erfi_x_np).float()
        self.erfi_values = torch.from_numpy(erfi_y_np).float()
        self.erfi_step = float(erfi_x_np[1] - erfi_x_np[0])

    def forward(self, params):
        """
        Predict diffusion MRI signals from NODDI parameters.

        params: (B, 7)
            [S0, fiso, ficvf, kappa, fibdir_x, fibdir_y, fibdir_z].

            This forward assumes params are physical parameters produced by
            output_to_param().

        return: (B, N)
            Predicted signal for B voxels and N diffusion measurements.
        """

        device = params.device
        dtype = params.dtype
        B = params.shape[0]

        grad_dirs = self.gradient.to(device=device, dtype=dtype)  # (N, 3)
        b = self.bval.to(device=device, dtype=dtype)

        S0 = params[:, 0]
        fiso = params[:, 1]
        ficvf = params[:, 2]
        kappa = params[:, 3].view(1, B)
        fibdir = params[:, 4:7]
        fibdir = fibdir / torch.linalg.norm(fibdir, dim=1, keepdim=True).clamp_min(1e-12)
        fibdir = fibdir.T.contiguous()  # (3, B), matching the old fibredir layout

        d_par = torch.full((1, B), self.di, device=device, dtype=dtype)
        d_iso = torch.full((1, B), self.diso, device=device, dtype=dtype)
        d_perp = d_par * (1.0 - ficvf.view(1, B))

        # —————————— Isotropic/free-water compartment ——————————
        E_iso = torch.exp(-b.view(-1, 1) @ d_iso)  # (N, B)

        # —————————— Extra-cellular hindered compartment ——————————
        d_w_par, d_w_perp = _watson_hindered_diffusion_coeff(
            d_par, d_perp, kappa, self.erfi_x, self.erfi_values, self.erfi_step
        )
        cos_theta = (grad_dirs @ fibdir).clamp(-1.0, 1.0)  # (N, B)
        cos2 = cos_theta.pow(2)
        E_hindered = torch.exp(
            -b.view(-1, 1) * ((d_w_par - d_w_perp) * cos2 + d_w_perp)
        )

        # —————————— Intra-cellular restricted/stick compartment ——————————
        lgi = _legendre_gaussian_integral(b.view(-1, 1) @ d_par, n=6)  # (N, 7, B)
        coeff = _watson_sh_coeff(kappa, self.erfi_x, self.erfi_values, self.erfi_step)  # (B, 7)
        sh_norm = torch.sqrt(
            (torch.arange(7, device=device, dtype=dtype) + 0.25) / math.pi
        ).view(1, 7, 1)
        sh = _legendre_even(cos_theta) * sh_norm  # (N, 7, B)

        E_restricted_raw = torch.sum(lgi * coeff.T.unsqueeze(0) * sh, dim=1)
        positive = E_restricted_raw > 0
        min_positive = torch.where(
            positive.any(),
            E_restricted_raw[positive].min().detach() * 0.1,
            E_restricted_raw.new_tensor(1e-12),
        )
        E_restricted = torch.where(positive, E_restricted_raw, min_positive)
        E_restricted = 0.5 * E_restricted

        # —————————— Tissue anisotropic signal ——————————
        ficvf_nb = ficvf.view(1, B)
        E_aniso = (1.0 - ficvf_nb) * E_hindered + ficvf_nb * E_restricted

        # —————————— Full NODDI signal ——————————
        fiso_nb = fiso.view(1, B)
        E_norm = ((1.0 - fiso_nb) * E_aniso + fiso_nb * E_iso).clamp(0.0, 1.0)
        return (E_norm * S0.view(1, B)).T.contiguous()
