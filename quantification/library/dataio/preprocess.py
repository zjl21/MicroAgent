def normalize_none(signal, bval, eps=1e-8):
    return signal, 1


def normalize_b0_mean(signal, bval, eps=1e-8):
    b0_mask = bval < 50
    b0 = signal[:, b0_mask].mean(dim=1, keepdim=True)
    b0_mean = b0.mean()
    return signal / b0_mean, b0_mean

def normalize_b0_median(signal, bval, eps=1e-8):
    b0_mask = bval < 50
    b0 = signal[:, b0_mask].mean(dim=1, keepdim=True)
    b0_median = b0.median()
    return signal / b0_median, b0_median


def normalize_max(signal, bval, eps=1e-8):
    b0_mask = bval < 50
    b0 = signal[:, b0_mask].mean(dim=1, keepdim=True)
    max = b0.max() + eps
    return signal / max, max