import os
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp

from torch import nn
from torchvision import transforms
from pathlib import Path
from dataclasses import dataclass
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.hybrid import Hybrid
from models.hybridSkip import Hybrid as Skip
from models.hybridDoubleSkip import Hybrid as DoubleSkip
from models.hybridNoPool import Hybrid as HybridNoPool
from models.pca import PCA_Net
from models.vae import VAE_Mixed
from models.unet import UNet
from utils.dataLoader import LandmarksDataset, ToTensor, Rescale
from utils.utils import scipy_to_torch_sparse, genMatrixesLH
from utils.fun import drawOrgans


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    n_nodes: List[int]
    latents: int = 64
    input_size: int = 1024
    filters: Optional[List[int]] = None
    skip_features: int = 32
    window: Tuple[int, int] = (3, 3)
    K: int = 6
    l1: Optional[int] = None
    l2: Optional[int] = None
    layer: Optional[int] = None
    extended: bool = False
    all_organs: bool = False
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    data_root: Path = Path(r"E:/Kai_2/DATA_Set/X-ray/HybridGNet/JSRT")
    weights_root: Path = Path(r"E:/Kai_2/Model-weights/HybridGNet/weights")

    def __post_init__(self):
        """Initialize derived attributes."""
        if self.filters is None:
            f = 32
            self.filters = [2, f, f, f, f//2, f//2, f//2]
        
        # Set up paths
        self.train_path = self.data_root / "Train"
        self.test_path = self.data_root / "Test"
        self.val_path = self.data_root / "Val"
        
        # Set up checkpoint paths
        self.pca_net_checkpoint = self.weights_root / "baselines/pca/best.pt"
        self.vae_mixed_checkpoint = self.weights_root / "baselines/vae/best.pt"
        self.hybrid_mixed_checkpoint = self.weights_root / "HybridGNet/best.pt"
        self.unet_checkpoint = self.weights_root / "UNet/best.pt"


class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        self.setup_data()
        self.setup_matrices()
        self.initialize_models()


    def setup_data(self):
        """Set up the test dataset."""
        self.test_dataset = LandmarksDataset(
            img_path=self.config.test_path / 'Images',
            label_path=self.config.test_path / 'landmarks',
            transform=transforms.Compose([
                Rescale(1024),
                ToTensor()
            ])
        )


    def setup_matrices(self):
        """Generate and prepare matrices for the models."""
        A, AD, D, U = genMatrixesLH()
        
        matrices = {
            'A': sp.csc_matrix(A).tocoo(),
            'AD': sp.csc_matrix(AD).tocoo(),
            'D': sp.csc_matrix(D).tocoo(),
            'U': sp.csc_matrix(U).tocoo()
        }
        
        D_ = [matrices['D'].copy()]
        U_ = [matrices['U'].copy()]
        A_ = [matrices['A'].copy()] * 3 + [matrices['AD'].copy()] * 3
        
        self.A_t, self.D_t, self.U_t = (
            [scipy_to_torch_sparse(x).to(self.device) for x in X]
            for X in (A_, D_, U_)
        )


    def load_model(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        """Load a model with its weights."""
        model.load_state_dict(torch.load(checkpoint_path))
        return model.eval().to(self.device)


    def initialize_models(self):
        """Initialize all models with their weights."""
        self.models = {
            'PCA': self.load_model(PCA_Net(self.config), self.config.pca_net_checkpoint),
            'VAE': self.load_model(VAE_Mixed(self.config), self.config.vae_mixed_checkpoint),
            'HybridGNet': self.load_model(
                Hybrid(self.config, self.D_t, self.U_t, self.A_t),
                self.config.hybrid_mixed_checkpoint
            ),
            'UNet': self.load_model(UNet(n_classes=3), self.config.unet_checkpoint)
        }
        
        # Initialize Skip models
        for layer in range(3, 7):
            self.config.layer = layer
            self.models[f'Skip{layer}'] = self.load_model(
                Skip(self.config, self.D_t, self.U_t, self.A_t),
                self.config.weights_root / f"Skip/skip_L{layer}/best.pt"
            )

        # Initialize Double Skip models
        for l1, l2 in [(4,3), (5,4), (6,5)]:
            self.config.l1, self.config.l2 = l1, l2
            self.models[f'Double{l1}{l2}'] = self.load_model(
                DoubleSkip(self.config, self.D_t, self.U_t, self.A_t),
                self.config.weights_root / f"Skip/double_L{l1}{l2}/best.pt"
            )


    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all models on the test dataset."""
        results = []
        
        for i, sample in enumerate(self.test_dataset):
            print(f'\r{i+1} of {len(self.test_dataset)}', end='')
            
            data = sample['image'].unsqueeze(0).to(self.device)
            target = sample['landmarks'][:120,:].reshape(-1).numpy()
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    output = model(data)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.cpu().numpy().reshape(-1)
                    print(model_name, target.shape, output.shape, "aaaaaaaaaaaaa")
                    mse = mean_squared_error(target * 1024, output * 1024)
                    results.append({
                        'sample_idx': i,
                        'MSE': mse,
                        'Model': model_name
                    })
        
        return pd.DataFrame(results)


    def visualize_results(self, results: pd.DataFrame, save_path: Optional[Path] = None):
        """Visualize evaluation results using boxplots."""
        plt.figure(figsize=(10,5))
        plt.tight_layout()
        sns.boxplot(x='Model', y='MSE', data=results, showmeans=True)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel('MSE')
        plt.title('Model Performance Comparison')
        plt.xlabel(None)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            
        # Print summary statistics
        print('\nMSE Statistics:')
        for model in results['Model'].unique():
            model_results = results[results['Model'] == model]['MSE']
            print(f"{model:15} {model_results.mean():.3f} ± {model_results.std():.3f}")


def main():
    """Main execution function."""
    config = ModelConfig(
        n_nodes=[120, 120, 120, 60, 60, 60],
        input_size=1024,
        skip_features=32
    )
    
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_models()
    evaluator.visualize_results(results, save_path=Path('figs/model_comparison.pdf'))


if __name__ == "__main__":
    main()