import torch.nn as nn

from .components import make_voxel_mlp


class MLP(nn.Module):
    """
    Voxel-wise DTI inversion network

    Uses the shared VoxelMLP or KANMLP decoder.
    MLP/backprop reference: https://www.nature.com/articles/323533a0
    KAN paper: https://arxiv.org/abs/2404.19756

    输入:
        signal: (B, N_grad)
    输出:
        params: (B, 7)
            [log S0, l11, l21, l22, l31, l32, l33]
    """

    def __init__(self, n_grad, hidden=128, num_layers=3, out_params=7, use_KAN: bool = False,
                 mlp_neck: int = 64, mlp_dropout: float = 0.0):
        super().__init__()
        self.net = make_voxel_mlp(
            n_grad,
            hidden,
            num_layers,
            out_params,
            neck_ch=mlp_neck,
            dropout=mlp_dropout,
            use_kan=use_KAN,
        )

    def forward(self, signal):
        """
        signal: (B, N_grad)
        """
        return self.net(signal)
