from dataclasses import dataclass
from typing import Tuple, List
import torch


@dataclass
class ModelConfig:
    n_nodes: List[int]
    latents: int = 64
    input_size: int = 1024
    filters: List[int] = None
    skip_features: int = 32
    window: Tuple[int, int] = (3, 3)
    K: int = 6
    l1: int = None
    l2: int = None
    layer: int = None
    extended: bool = False
    all_organs: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # train_path: str = "../Datasets/JSRT/Train"
    # test_path: str = "../Datasets/JSRT/Test"
    # val_path: str = "../Datasets/JSRT/Val" 

    # pca_net_checkpoint: str = '../weights/baselines/pca/best.pt'
    # vae_mixed_checkpoint: str = '../weights/baselines/vae/best.pt'
    # hybrid_mixed_checkpoint: str = '../weights/HybridGNet/best.pt'
    # unet_checkpoint: str = '../weights/UNet/best.pt'

    train_path: str = "E:\Kai_2\DATA_Set\X-ray\HybridGNet\JSRT\Train"
    test_path: str = "E:\Kai_2\DATA_Set\X-ray\HybridGNet\JSRT\Test"
    val_path: str = "E:\Kai_2\DATA_Set\X-ray\HybridGNet\JSRT\Val" 

    pca_net_checkpoint: str = r"E:\Kai_2\Model-weights\HybridGNet\weights\baselines\pca\best.pt"
    vae_mixed_checkpoint: str = r"E:\Kai_2\Model-weights\HybridGNet\weights\baselines\vae\best.pt"
    hybrid_mixed_checkpoint: str = r"E:\Kai_2\Model-weights\HybridGNet\weights\HybridGNet\best.pt"
    unet_checkpoint: str = r"E:\Kai_2\Model-weights\HybridGNet\weights\UNet\best.pt"

    def __post_init__(self):
        if self.filters is None:
            f = 32
            self.filters = [2, f, f, f, f//2, f//2, f//2]