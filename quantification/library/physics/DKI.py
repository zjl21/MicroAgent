import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
import torch
from glob import glob


def _require_command(name: str, env_var: str) -> str:
    """Resolve an imaging command from an override or ``PATH``."""
    command = os.environ.get(env_var) or shutil.which(name)
    if not command:
        raise FileNotFoundError(
            f"Required command '{name}' was not found. Put it on PATH or set "
            f"{env_var} to the executable path."
        )
    command = os.path.abspath(os.path.expanduser(command))
    if not os.path.isfile(command) or not os.access(command, os.X_OK):
        raise FileNotFoundError(f"{env_var} does not name an executable: {command}")
    return command

def get_gradient(bval, bvec):
    bval = np.asarray(bval)/1000  # (N,)
    bvec = np.asarray(bvec)  # (3, N)

    D_ind = np.array(
        [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
    )
    D_cnt = np.array([1, 1, 1, 2, 2, 2])[:, None]

    W_ind = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [1, 1, 1, 2],
            [1, 2, 2, 2],
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [0, 1, 2, 2],
        ]
    )
    W_cnt = np.array([1, 1, 1, 4, 4, 4, 4, 4, 4, 6, 6, 6, 12, 12, 12])[:, None]

    bD = D_cnt * bvec[D_ind[:, 0], :] * bvec[D_ind[:, 1], :]
    bW = (
        W_cnt
        * bvec[W_ind[:, 0], :]
        * bvec[W_ind[:, 1], :]
        * bvec[W_ind[:, 2], :]
        * bvec[W_ind[:, 3], :]
    )

    b_row = bval[None, :]
    gradient = np.concatenate([
        -np.repeat(b_row, 6, axis=0) * bD,
        (np.repeat(b_row, 15, axis=0) ** 2) / 6.0 * bW,
    ], axis=0)  # (21, N)
    return gradient

def constrain(params):
    fpConstrain = os.path.join(os.path.dirname(__file__), 'DKI_kurtosis_constrain.txt')
    kurtosis_constrain = torch.tensor(np.loadtxt(fpConstrain)).to(params.device)  # (N, 15)

    B = params.shape[0]
    N = kurtosis_constrain.shape[1]

    K_param = params[:, 7:] # (B, 15)
    g = kurtosis_constrain.T.unsqueeze(0).expand(B, N, 15).to(dtype=K_param.dtype)  # (B, N, 15)
    exponent = torch.matmul(g, K_param.unsqueeze(2)).squeeze(2)  # (B, N)

    return torch.mean(torch.relu(-exponent))

class DKI:
    """
    网络输出：S0 + Cholesky 分量（L11, L21, L22, L31, L32, L33） + kurtosis 分量
    （W1111, W2222, W3333, W1112, W1113, W1222, W1333, W2223, W2333, W1122, W1133, W2233, W1123, W1223, W1233）(W是乘上 (1/3(Dxx+Dyy+Dzz))^2 的版本)
    """
    
    require_S0 = False
    PARAM_NAMES = ["S0", "L11", "L21", "L22", "L31", "L32", "L33",
                   "W1111", "W2222", "W3333", "W1112", "W1113", "W1222", "W1333", "W2223", "W2333", "W1122", "W1133", "W2233", "W1123", "W1223", "W1233"]
    N_PARAMS    = len(PARAM_NAMES)   # 7 + 15 = 22

    @classmethod
    def recon_signal(cls, dpOut: str, dpDiff: str):
        """
        根据网络输出的参数文件和输入数据文件，重建预测的信号。
        """
        fpBval = glob(os.path.join(dpDiff, '*_diff.bval'))[0]
        fpBvec = glob(os.path.join(dpDiff, '*_diff.bvec'))[0]
        fpDiff = glob(os.path.join(dpDiff, '*_diff.nii.gz'))[0]

        b      = np.loadtxt(fpBval)  # (N,)
        bvec   = np.loadtxt(fpBvec)  # (3, N)
        ref    = nib.load(fpDiff)
        if bvec.shape[0] != 3:
            bvec = bvec.T

        gradient = get_gradient(b, bvec)  # (21, N)

        fpTensor = os.path.join(dpOut, "tensor.nii.gz")
        fpKurtosis = os.path.join(dpOut, "kurtosis.nii.gz")
        fpS0 = os.path.join(dpOut, "S0.nii.gz")
        tensor = nib.load(fpTensor).get_fdata()
        kurtosis = nib.load(fpKurtosis).get_fdata()
        MD = (tensor[..., 0] + tensor[..., 1] + tensor[..., 2]) / 3
        kurtosis = kurtosis * (MD[..., np.newaxis] ** 2)

        DK = np.concatenate([tensor, kurtosis], axis=3)  # (B, 21)
        S0 = nib.load(fpS0).get_fdata() 
        
        dot_product = np.tensordot(DK, gradient, axes=([3], [0]))
        diff = S0[..., np.newaxis] * np.exp(dot_product)
        diff_img = nib.Nifti1Image(diff.astype(np.float32), ref.affine, ref.header)
        fpDiffPred = os.path.join(dpOut, "diff_pred.nii.gz")
        nib.save(diff_img, fpDiffPred)
        print(f"✅ 已保存重建的扩散信号到 {fpDiffPred}")
    

    @classmethod
    def output_to_param(
        cls, 
        output: torch.Tensor, 
    ) -> torch.Tensor:
        
        # if output_normalize:
        #     # S0
        #     S0 = torch.relu(output[..., 0])

        #     LOWER = torch.tensor([
        #         0, -0.8, 0, 
        #         -0.8, -0.7, 0,
        #         0, 0, 0,
        #         -0.7, -0.7, -0.7,
        #         -0.7, -0.7, -0.7,
        #         -0.1, -0.1, -0.1,
        #         -0.3, -0.3, -0.3,
        #     ], device=output.device, dtype=output.dtype)

        #     UPPER = torch.tensor([
        #         2, 1, 2,
        #         0.7, 0.7, 2,
        #         5.5, 5.5, 5.5,
        #         0.7, 0.7, 0.7,
        #         0.7, 0.7, 0.7,
        #         2, 2, 2,
        #         0.3, 0.3, 0.3,
        #     ], device=output.device, dtype=output.dtype)
        #     cholesky = LOWER[:6] + torch.sigmoid(output[..., 1:7]) * (UPPER[:6] - LOWER[:6])
        #     cholesky = torch.cat([S0.unsqueeze(-1), cholesky], dim=-1)
        #     W = LOWER[6:] + torch.sigmoid(output[..., 7:]) * (UPPER[6:] - LOWER[6:])

        # else:
        cholesky = output[..., :7]
        W = output[..., 7:]

        S0 = torch.relu(cholesky[..., 0:1])
        L11 = cholesky[..., 1]
        L21 = cholesky[..., 2]
        L22 = cholesky[..., 3]
        L31 = cholesky[..., 4]
        L32 = cholesky[..., 5]
        L33 = cholesky[..., 6]

        Dxx = L11 ** 2
        Dyy = L21 ** 2 + L22 ** 2
        Dzz = L31 ** 2 + L32 ** 2 + L33 ** 2
        Dxy = L11 * L21
        Dxz = L11 * L31
        Dyz = L21 * L31 + L22 * L32

        tensor = torch.stack([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], dim=-1)
        MD = (Dxx + Dyy + Dzz) / 3
        kurtosis = W / (MD.unsqueeze(-1) ** 2 + 1e-8)

        return torch.cat([S0, tensor, kurtosis], dim=-1)

    @classmethod
    def save_nifti(cls, params_all: np.ndarray, mask: np.ndarray,
                   affine: np.ndarray, out_dir: str, norm_factor=1.0):
        """
        将模型输出的原始参数反归一化后写回 3D NIfTI。

        params_all  : (N_voxels, n_params) numpy array，模型直接输出
        mask        : (H, W, Z) bool array
        affine      : NIfTI affine matrix
        out_dir     : 输出目录
        norm_factor : 预处理时的归一化因子（标量或 tensor）
        """
        if hasattr(norm_factor, "item"):
            norm_factor = norm_factor.item()

        os.makedirs(out_dir, exist_ok=True)
        H, W, Z   = mask.shape
        flat_mask = mask.reshape(-1)

        params_out = params_all.copy()
        params_out[:, 0] *= norm_factor   # S0 反归一化

        # 保存S0
        S0 = np.zeros(H * W * Z, dtype=np.float32)
        S0[flat_mask] = params_out[:, 0]
        fpS0 = os.path.join(out_dir, "S0.nii.gz")
        nib.save(nib.Nifti1Image(S0.reshape(H, W, Z), affine), fpS0)

        # 保存 D 参数（MRtrix3格式）
        D = np.zeros((H * W * Z, 6), dtype=np.float32)
        D[flat_mask] = params_out[:, 1:7]
        fpD = os.path.join(out_dir, "tensor.nii.gz")
        nib.save(nib.Nifti1Image(D.reshape(H, W, Z, 6), affine), fpD)

        fpFA = os.path.join(out_dir, "FA.nii.gz")
        fpMD = os.path.join(out_dir, "MD.nii.gz")
        fpRD = os.path.join(out_dir, "RD.nii.gz")
        fpL1 = os.path.join(out_dir, "L1.nii.gz")
        fpL2 = os.path.join(out_dir, "L2.nii.gz")
        fpL3 = os.path.join(out_dir, "L3.nii.gz")
        fpValue = os.path.join(out_dir, "value.nii.gz")
        fpV1 = os.path.join(out_dir, "V1.nii.gz")
        fpV2 = os.path.join(out_dir, "V2.nii.gz")
        fpV3 = os.path.join(out_dir, "V3.nii.gz")
        fpVector = os.path.join(out_dir, "vector.nii.gz")

        # 保存 K 参数（MRtrix3格式）
        K = np.zeros((H * W * Z, 15), dtype=np.float32)
        K[flat_mask] = params_out[:, 7:]
        fpK = os.path.join(out_dir, "kurtosis.nii.gz")
        nib.save(nib.Nifti1Image(K.reshape(H, W, Z, 15), affine), fpK)
        fpAK = os.path.join(out_dir, "AK.nii.gz")
        fpRK = os.path.join(out_dir, "RK.nii.gz")
        fpMK = os.path.join(out_dir, "MK.nii.gz")

        tensor2metric = _require_command(
            "tensor2metric", "MICROAGENT_TENSOR2METRIC"
        )
        fslroi = _require_command("fslroi", "MICROAGENT_FSLROI")
        subprocess.run(
            [
                tensor2metric, fpD,
                "-dkt", fpK,
                "-fa", fpFA,
                "-adc", fpMD,
                "-rd", fpRD,
                "-value", fpValue,
                "-vector", fpVector,
                "-num", "1,2,3",
                "-ak", fpAK,
                "-rk", fpRK,
                "-mk", fpMK,
                "-force", "-quiet",
            ],
            check=True,
        )
        for source, target, start, size in (
            (fpValue, fpL1, 0, 1),
            (fpValue, fpL2, 1, 1),
            (fpValue, fpL3, 2, 1),
            (fpVector, fpV1, 0, 3),
            (fpVector, fpV2, 3, 3),
            (fpVector, fpV3, 6, 3),
        ):
            subprocess.run(
                [fslroi, source, target, str(start), str(size)],
                check=True,
            )

    def __init__(self, bval, bvec, norm_factor = 1.0):
        """
        bval: (N,)
        bvec: (3, N)
        """
        if bvec.shape[0] != 3:
            bvec = bvec.T

        self.gradient = torch.tensor(get_gradient(bval, bvec)).float()  # (21, N)
        self.norm_factor = norm_factor

    def forward(self, params):
        """
        params: (B, 22)
            [S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz,
            (W1111, W2222, W3333, W1112, W1113, W1222, W1333, W2223, W2333, W1122, W1133, W2233, W1123, W1223, W1233) * (1/3(Dxx+Dyy+Dzz))^2] 
            L 为 Cholesky 因子，单位 µm/ms^0.5（D = LL^T 单位为 µm²/ms, W 为 kurtosis 分量）

        return:
            signal_pred: (B, N)
        """
        B = params.shape[0]
        N = self.gradient.shape[1]

        S0  = params[:, 0]
        D_param = params[:, 1:7]  # (B, 6)
        MD = D_param[:, :3].mean(dim=1, keepdim=True)  # (B, 1)
        W_param = params[:, 7:] * (MD ** 2)
        DK = torch.cat([D_param, W_param], dim=1)  # (B, 21)

        g = self.gradient.T.unsqueeze(0).expand(B, N, 21)  # (B, N, 21)
        exponent = torch.matmul(g, DK.unsqueeze(2)).squeeze(2)  # (B, N)
        exponent = torch.clamp(exponent, max=20.0)
        signal_pred = S0.unsqueeze(1) * torch.exp(exponent)  # (B, N)

        return signal_pred
