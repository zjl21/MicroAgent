import nibabel as nib
import numpy as np
from scipy import fftpack
import bm4d
from bm4d.profiles import BM4DStages
from threadpoolctl import threadpool_limits

# NumPy >=2 removed np.trapz; bm4d still calls it internally.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

def denoise(data, cpu_limit=8):
    # 归一化到 [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min < 1e-8:
        return data
    data_norm = (data - data_min) / (data_max - data_min)
    
    nx, ny, nz = data_norm.shape
    psd_acc = np.zeros((nx//2, ny//2))
    
    # 计算平均 PSD (Power Spectral Density)
    for z in range(nz):
        slice_2d = data_norm[:, :, z]
        f2d = fftpack.fft2(slice_2d)
        f2d_shifted = fftpack.fftshift(f2d)
        psd_2d = np.abs(f2d_shifted) ** 2
        psd_acc += psd_2d[:nx//2, :ny//2]
    
    psd_avg = psd_acc / nz
    
    # BM4D 去噪
    with threadpool_limits(limits=cpu_limit):
        y_hat = bm4d.bm4d(data_norm, psd_avg,  stage_arg=BM4DStages.HARD_THRESHOLDING)
    return y_hat * (data_max - data_min) + data_min  # 恢复原始范围


class BM4D:
    def __init__(self, input_file, output_file, mask_file):
        self.input_file = input_file
        self.output_file = output_file
        self.mask_file = mask_file

    def run(self):
        noisy_img = nib.load(self.input_file)
        noisy_data = noisy_img.get_fdata()
        mask = nib.load(self.mask_file).get_fdata()
        mask = np.expand_dims(mask,axis=3)
        noisy_data = noisy_data * mask  # 应用掩码
        clean_data = np.zeros_like(noisy_data)
        for c in range(noisy_data.shape[3]):
            volume = noisy_data[:, :, :, c]
            denoised_volume = denoise(volume)
            clean_data[:, :, :, c] = denoised_volume
        clean_data = clean_data * mask

        clean_img = nib.Nifti1Image(clean_data, affine=noisy_img.affine, header=noisy_img.header)
        nib.save(clean_img, self.output_file)
