import os
import sys
import torch

from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from model_loader import ModelLoader
from models.unet import UNet
from models.pca import PCA_Net
from models.vae import VAE_Mixed
from models.hybrid import Hybrid
from evaluation import ModelEvaluator
from visualization import plot_model_comparison
from utils.dataLoader import LandmarksDataset, ToTensor, Rescale


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


def main():
    config = ModelConfig(
        n_nodes=[120, 120, 120, 60, 60, 60],
        input_size=1024,
        skip_features=32
    )
    
    # Data setup
    test_dataset = LandmarksDataset(
        img_path=os.path.join(config.test_path, 'Images'),
        label_path=os.path.join(config.test_path, 'landmarks'),
        transform=transforms.Compose([
            Rescale(1024),
            ToTensor()
        ])
    )
    
    # Model initialization
    loader = ModelLoader(config, config.device)
    models = {
        'PCA': loader.load_model(PCA_Net, config.pca_net_checkpoint),
        'VAE': loader.load_model(VAE_Mixed, config.vae_mixed_checkpoint),
        'HybridGNet': loader.load_model(Hybrid, config.hybrid_mixed_checkpoint),
        'UNet': loader.load_model(UNet, config.unet_checkpoint, n_classes=3)
    }
    
    # Evaluation
    evaluator = ModelEvaluator(test_dataset, config.device)
    results = evaluator.evaluate_models(
        list(models.values()),
        list(models.keys())
    )
    
    # Visualization
    plot_model_comparison(results, list(models.keys()), 'figs/model_comparison.pdf')

if __name__ == '__main__':
    main()