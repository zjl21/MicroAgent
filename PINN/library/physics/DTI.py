import os
import numpy as np
import nibabel as nib
import torch
from glob import glob


def _tensor_metrics(tensor: np.ndarray):
    """Compute DTI eigenvalues, eigenvectors, FA, MD, and RD with NumPy.

    ``tensor`` uses the MRtrix component order
    ``[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]``. Eigenvalues and the corresponding
    eigenvectors are returned in descending order.
    """
    matrix = np.empty((tensor.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = tensor[:, 0]
    matrix[:, 1, 1] = tensor[:, 1]
    matrix[:, 2, 2] = tensor[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = tensor[:, 3]
    matrix[:, 0, 2] = matrix[:, 2, 0] = tensor[:, 4]
    matrix[:, 1, 2] = matrix[:, 2, 1] = tensor[:, 5]

    values, vectors = np.linalg.eigh(matrix)
    values = values[:, ::-1]
    vectors = vectors[:, :, ::-1]
    md = values.mean(axis=1)
    rd = values[:, 1:].mean(axis=1)
    numerator = 1.5 * np.square(values - md[:, None]).sum(axis=1)
    denominator = np.square(values).sum(axis=1)
    fa = np.sqrt(np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    ))
    return values.astype(np.float32), vectors.astype(np.float32), fa.astype(np.float32), md.astype(np.float32), rd.astype(np.float32)

class DTI:
    # 网络输出 Cholesky 分量：S0 + L11, L21, L22, L31, L32, L33
    # D = L L^T 保证正定
    
    require_S0 = False
    PARAM_NAMES = ["S0", "L11", "L21", "L22", "L31", "L32", "L33"]
    N_PARAMS    = len(PARAM_NAMES)   # 7

    @classmethod
    def recon_signal(cls, dpOut: str, dpDiff: str):
        """
        根据网络输出的参数文件和输入数据文件，重建预测的信号。
        """
        fpBval = glob(os.path.join(dpDiff, '*_diff.bval'))[0]
        fpBvec = glob(os.path.join(dpDiff, '*_diff.bvec'))[0]
        fpDiff = glob(os.path.join(dpDiff, '*_diff.nii.gz'))[0]

        b      = np.loadtxt(fpBval) / 1000  # (N,)
        bvec   = np.loadtxt(fpBvec)  # (3, N)
        ref    = nib.load(fpDiff)
        if bvec.shape[0] != 3:
            bvec = bvec.T

        gx = bvec[0,:]
        gy = bvec[1,:]
        gz = bvec[2,:]
        gradient = np.stack([b*gx*gx, b*gy*gy, b*gz*gz, 2*b*gx*gy, 2*b*gx*gz, 2*b*gy*gz])  # (6, N)

        fpTensor = os.path.join(dpOut, "tensor.nii.gz")
        fpS0 = os.path.join(dpOut, "S0.nii.gz")
        tensor = nib.load(fpTensor).get_fdata()
        S0 = nib.load(fpS0).get_fdata()

        dot_product = np.tensordot(tensor, gradient, axes=([3], [0]))
        diff = S0[..., np.newaxis] * np.exp(-dot_product)
        diff_img = nib.Nifti1Image(diff.astype(np.float32), ref.affine, ref.header)
        fpDiffPred = os.path.join(dpOut, "diff_pred.nii.gz")
        nib.save(diff_img, fpDiffPred)
        print(f"✅ 已保存重建的扩散信号到 {fpDiffPred}")

    @classmethod
    def output_to_param(
        cls,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert network output to physical DTI parameters:
        [..., S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
        """
        # if output_normalize:
        #     S0 = torch.relu(output[..., 0])

        #     LOWER = torch.tensor([
        #         0.3, -0.7, 0.2,
        #         -0.7, -0.6, 0.2
        #     ], device=output.device, dtype=output.dtype)

        #     UPPER = torch.tensor([
        #         2, 0.7, 2,
        #         0.7, 0.6, 2
        #     ], device=output.device, dtype=output.dtype)

        #     cholesky = LOWER + (UPPER - LOWER) * torch.sigmoid(output[..., 1:7])
        #     cholesky = torch.concat([S0.unsqueeze(-1), cholesky], dim=-1)
        # else:
        cholesky = output

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
        return torch.cat([S0, tensor], dim=-1)
        

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

        values_brain, vectors_brain, fa_brain, md_brain, rd_brain = _tensor_metrics(
            params_out[:, 1:7]
        )

        def save_scalar(name, brain_values):
            volume = np.zeros(H * W * Z, dtype=np.float32)
            volume[flat_mask] = brain_values
            nib.save(
                nib.Nifti1Image(volume.reshape(H, W, Z), affine),
                os.path.join(out_dir, f"{name}.nii.gz"),
            )

        def save_vector(name, brain_vectors):
            volume = np.zeros((H * W * Z, 3), dtype=np.float32)
            volume[flat_mask] = brain_vectors
            nib.save(
                nib.Nifti1Image(volume.reshape(H, W, Z, 3), affine),
                os.path.join(out_dir, f"{name}.nii.gz"),
            )

        save_scalar("FA", fa_brain)
        save_scalar("MD", md_brain)
        save_scalar("RD", rd_brain)
        for index, name in enumerate(("L1", "L2", "L3")):
            save_scalar(name, values_brain[:, index])
        for index, name in enumerate(("V1", "V2", "V3")):
            save_vector(name, vectors_brain[:, :, index])

        value = np.zeros((H * W * Z, 3), dtype=np.float32)
        value[flat_mask] = values_brain
        nib.save(
            nib.Nifti1Image(value.reshape(H, W, Z, 3), affine),
            os.path.join(out_dir, "value.nii.gz"),
        )
        vector = np.zeros((H * W * Z, 9), dtype=np.float32)
        vector[flat_mask] = vectors_brain.transpose(0, 2, 1).reshape(-1, 9)
        nib.save(
            nib.Nifti1Image(vector.reshape(H, W, Z, 9), affine),
            os.path.join(out_dir, "vector.nii.gz"),
        )

    def __init__(self, bval, bvec, norm_factor=1.0):
        """
        bval: (N,)
        bvec: (3, N)
        """
        b = bval / 1000  # 转换为 ms/um²
        gx = bvec[0,:]
        gy = bvec[1,:]
        gz = bvec[2,:]
        self.gradient = torch.stack([b*gx*gx, b*gy*gy, b*gz*gz, 2*b*gx*gy, 2*b*gx*gz, 2*b*gy*gz])  # (6, N)
        self.norm_factor = norm_factor

    def forward(self, params):
        """
        params: (B, 7)
            [S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], tensor in um^2/ms

        return:
            signal_pred: (B, N)
        """

        B = params.shape[0]
        N = self.gradient.shape[1]

        S0 = params[:, 0]
        D_param = params[:, 1:7]

        g = self.gradient.T.unsqueeze(0).expand(B, N, 6)  # (B, N, 6)
        exponent = -torch.matmul(g, D_param.unsqueeze(2)).squeeze(2)  # (B, N)
        exponent = torch.clamp(exponent, max=20.0)
        signal_pred = S0.unsqueeze(1) * torch.exp(exponent)  # (B, N)

        return signal_pred
