"""
Medical image analysis and visualization using PyTorch.
Handles model evaluation, error analysis, and visualization of medical imaging predictions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

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

config = ModelConfig(
    n_nodes=[120, 120, 120, 120, 120, 120],
    input_size=1024,
    skip_features=32
)


@dataclass
class ModelConfig:
    """Configuration for model evaluation and visualization."""
    device: torch.device
    output_dir: Path
    image_size: int = 1024
    landmarks_dim: int = 120

class ModelEvaluator:
    """Handles evaluation of multiple models on medical image data."""
    
    def __init__(
        self, 
        models: List[nn.Module],
        model_names: List[str],
        config: ModelConfig
    ):
        self.models = models
        self.model_names = model_names
        self.config = config
        self.results_df = pd.DataFrame()


    def evaluate_single_sample(self, data: torch.Tensor, target: torch.Tensor) -> pd.DataFrame:
        """Evaluate all models on a single sample."""
        results = []
        
        data = data.unsqueeze(0).to(self.config.device)
        target = target[:self.config.landmarks_dim].reshape(-1).numpy()
        
        for model, model_name in zip(self.models, self.model_names):
            with torch.no_grad():
                output = model(data)
                output = output[0] if isinstance(output, tuple) else output
                output = output.cpu().numpy().reshape(-1)
                
                error = mean_squared_error(
                    target * self.config.image_size, 
                    output * self.config.image_size
                )
                
                results.append({
                    'sample_idx': len(self.results_df),
                    'MSE': error,
                    'Model': model_name
                })
                
        return pd.DataFrame(results)


    def evaluate_dataset(self, test_dataset) -> pd.DataFrame:
        """Evaluate all models on entire test dataset."""
        for i, sample in enumerate(test_dataset):
            print(f'\rProcessing sample {i+1} of {len(test_dataset)}', end='')
            sample_results = self.evaluate_single_sample(
                sample['image'], 
                sample['landmarks']
            )
            self.results_df = pd.concat(
                [self.results_df, sample_results], 
                ignore_index=True
            )
        print()  # New line after progress
        return self.results_df


    def evaluate_multiatlas(self, atlas_dir: Path, test_dir: Path) -> pd.DataFrame:
        """Evaluate MultiAtlas predictions."""

        def natural_sort_key(path_str: str) -> List[Union[int, str]]:
            """Key for natural sorting of paths."""
            return [
                int(s) if s.isdigit() else s 
                for s in re.split(r'(\d+)', path_str)
            ]

        atlas_files = sorted(
            [str(p) for p in atlas_dir.glob('*.npy')],
            key=natural_sort_key
        )
        
        results = []
        for i, atlas_file in enumerate(atlas_files):
            atlas_data = np.load(atlas_file)[:240]
            test_file = Path(atlas_file).name
            test_data = np.load(test_dir / test_file)[:240]
            
            error = mean_squared_error(test_data, atlas_data)
            results.append({
                'sample_idx': i,
                'MSE': error,
                'Model': 'MultiAtlas'
            })
            
        atlas_results = pd.DataFrame(results)
        self.results_df = pd.concat(
            [self.results_df, atlas_results], 
            ignore_index=True
        )
        return self.results_df


class ResultVisualizer:
    """Handles visualization of model evaluation results."""
    
    @staticmethod
    def plot_error_distribution(results_df: pd.DataFrame, output_path: Optional[Path] = None):
        """Plot box plot of MSE distribution for each model."""
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        
        sns.boxplot(
            x='Model',
            y='MSE',
            data=results_df,
            showmeans=True
        )
        
        plt.xticks(rotation=25, ha="right")
        plt.ylabel('MSE')
        plt.title('MSE Distribution by Model')
        plt.xlabel(None)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')


    @staticmethod
    def print_model_stats(results_df: pd.DataFrame, model_names: List[str]):
        """Print mean and std of MSE for each model."""
        print('MSE Statistics:')
        for model in model_names:
            model_results = results_df[results_df['Model'] == model]['MSE']
            print(f'{model:<20} {model_results.mean():.3f} ± {model_results.std():.3f}')


    @staticmethod
    def visualize_predictions(
        test_dataset,
        models: List[nn.Module],
        model_names: List[str],
        sample_indices: List[int],
        config: ModelConfig,
        draw_organs_fn,
        unet_model: Optional[nn.Module] = None
    ):
        """Create visualization grid of model predictions."""
        n_cols = len(models) + 3  # models + GT + MultiAtlas + UNet
        n_rows = len(sample_indices)
        
        fig = plt.figure(figsize=(24, 8), dpi=200)
        
        for row, sample_idx in enumerate(sample_indices):
            with torch.no_grad():
                sample = test_dataset[sample_idx]
                data = sample['image'].unsqueeze(0).to(config.device)
                target = sample['landmarks'].reshape(-1).numpy()
                image = data.cpu().numpy()[0, 0]
                
                # Ground Truth
                ax = plt.subplot(n_rows, n_cols, 1 + row * n_cols)
                plt.axis('off')
                ResultVisualizer._setup_subplot(ax, image, target[:240] * config.image_size, 
                                              draw_organs_fn)
                if row == 0:
                    plt.title("Ground Truth", fontsize=20)
                    
                # MultiAtlas
                ax = plt.subplot(n_rows, n_cols, 2 + row * n_cols)
                data_ma = ResultVisualizer._load_multiatlas_data(sample_idx, config)
                plt.axis('off')
                ResultVisualizer._setup_subplot(ax, image, data_ma, draw_organs_fn)
                if row == 0:
                    plt.title("MultiAtlas", fontsize=20)
                    
                # Model Predictions
                for col, (model, name) in enumerate(zip(models, model_names)):
                    output = model(data)
                    output = output[0] if isinstance(output, tuple) else output
                    output = np.clip(
                        output.cpu().numpy().reshape(-1)[:240], 
                        0, 1
                    ) * config.image_size
                    
                    ax = plt.subplot(n_rows, n_cols, col + 3 + row * n_cols)
                    plt.axis('off')
                    ResultVisualizer._setup_subplot(ax, image, output, draw_organs_fn)
                    if row == 0:
                        plt.title(name, fontsize=20)
                        
                # UNet (if provided)
                if unet_model is not None:
                    ax = plt.subplot(n_rows, n_cols, n_cols + row * n_cols)
                    plt.axis('off')
                    ResultVisualizer._visualize_unet(ax, unet_model, data, image)
                    if row == 0:
                        plt.title('UNet', fontsize=20)
        
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0)
        return fig


    @staticmethod
    def _setup_subplot(ax, image, landmarks, draw_organs_fn):
        """Setup individual subplot with consistent formatting."""
        plt.xlim(1, 1024)
        plt.ylim(1024, 1)
        draw_organs_fn(ax, landmarks, None, image.copy())


    @staticmethod
    def _load_multiatlas_data(sample_idx: int, config: ModelConfig) -> np.ndarray:
        """Load MultiAtlas prediction for a sample."""
        # Implementation depends on your data structure
        pass


    @staticmethod
    def _visualize_unet(ax, unet_model: nn.Module, data: torch.Tensor,base_image: np.ndarray):
        """Create UNet visualization overlay."""
        with torch.no_grad():
            output = unet_model(data)
            output = torch.argmax(output[0], dim=0).cpu().numpy()
            
            image = np.zeros((*base_image.shape, 3))
            image[:, :, 0] = base_image + 0.7 * (output == 1) - 0.3 * (output == 2)
            image[:, :, 1] = base_image + 0.7 * (output == 2) - 0.2 * (output == 1)
            image[:, :, 2] = base_image - 0.2 * (output == 1) - 0.3 * (output == 2)
            
            plt.imshow(np.clip(image, 0, 1))
            plt.xlim(1, 1024)
            plt.ylim(1024, 1)


# Example usage:
def main():
    config = ModelConfig(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir=Path('output')
    )
    
    # Initialize models and names
    models = [pcaNet, vae, hybrid, Skip6, double65]  # Your model instances
    model_names = ['PCA', 'FC', 'HybridGNet', '1-IGSC Layer 6', '2-IGSC Layers 6-5']
    
    # Create evaluator
    evaluator = ModelEvaluator(models, model_names, config)
    
    # Evaluate models
    results = evaluator.evaluate_dataset(test_dataset)
    
    # Add MultiAtlas results
    results = evaluator.evaluate_multiatlas(
        atlas_dir=Path("MultiAtlas/JSRT/images/output_points"),
        test_dir=Path("../Datasets/JSRT/Test/landmarks")
    )
    
    # Visualize results
    visualizer = ResultVisualizer()
    visualizer.plot_error_distribution(results)
    visualizer.print_model_stats(results, model_names + ['MultiAtlas'])
    
    # Visualize predictions
    worst_cases = results[results['Model'] == '2-SC Layers 6-5'].nlargest(3, 'MSE')
    fig = visualizer.visualize_predictions(
        test_dataset=test_dataset,
        models=models,
        model_names=model_names,
        sample_indices=worst_cases.index.tolist(),
        config=config,
        draw_organs_fn=drawOrgans,
        unet_model=modelUNet
    )
    
    # Save visualizations
    fig.savefig('compare_UNet.png', bbox_inches='tight', dpi=200)
    fig.savefig('compare_UNet.pdf', bbox_inches='tight', dpi=200)

if __name__ == '__main__':
    main()