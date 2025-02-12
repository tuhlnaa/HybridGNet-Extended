import os
import sys 
import torch
import numpy as np
import scipy.sparse as sp

from pathlib import Path
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.hybrid import Hybrid as Hybrid
from models.hybridSkip import Hybrid as Skip
from models.hybridDoubleSkip import Hybrid as DoubleSkip
from models.hybridNoPool import Hybrid as HybridNoPool
from models.pca import PCA_Net
from models.vae import VAE_Mixed
from models.unet import UNet

from utils.dataLoader import LandmarksDataset, ToTensor, Rescale
from utils.utils import scipy_to_torch_sparse, genMatrixesLH


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
    
    # Data paths
    data_root: Path = Path(r"E:/Kai_2/DATA_Set/X-ray/HybridGNet/JSRT")
    weights_root: Path = Path(r"E:/Kai_2/Model-weights/HybridGNet")

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

# Data setup
test_dataset = LandmarksDataset(
    img_path=os.path.join(config.test_path, 'Images'),
    label_path=os.path.join(config.test_path, 'landmarks'),
    transform=transforms.Compose([
        Rescale(1024),
        ToTensor()
    ])
)


A, AD, D, U = genMatrixesLH()

matrices = {
    'A': sp.csc_matrix(A).tocoo(),
    'AD': sp.csc_matrix(AD).tocoo(),
    'D': sp.csc_matrix(D).tocoo(),
    'U': sp.csc_matrix(U).tocoo()
}

D_ = [matrices['D'].copy()]
U_ = [matrices['U'].copy()]
A_ = [matrices['A'].copy()] * 6

A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(config.device) for x in X] for X in (A_, D_, U_))


hybridNP = HybridNoPool(config, D_t, U_t, A_t).to(config.device)
hybridNP.load_state_dict(torch.load(config.weights_root / "hybrid_no_pool/bestMSE_2000.pt"))
hybridNP.eval()

# =========================================================
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

A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(config.device) for x in X] for X in (A_, D_, U_))

# =========================================================
config.n_nodes = [120, 120, 120, 60, 60, 60]
config.l1 = 5
config.l2 = 4

double54 = DoubleSkip(config, D_t, U_t, A_t).to(config.device)
double54.load_state_dict(torch.load(config.weights_root / "weights/Skip/double_L54/best.pt"))
double54.eval()

# =========================================================
config.l1 = 6
config.l2 = 5

double65 = DoubleSkip(config, D_t, U_t, A_t).to(config.device)
double65.load_state_dict(torch.load(config.weights_root / "weights/Skip/double_L65/best.pt"))
double65.eval()

# =========================================================
config.l1 = 4
config.l2 = 3

double43 = DoubleSkip(config, D_t, U_t, A_t).to(config.device)
double43.load_state_dict(torch.load(config.weights_root / "weights/Skip/double_L43/best.pt"))
double43.eval()

hybrid = Hybrid(config, D_t, U_t, A_t).to(config.device)
hybrid.load_state_dict(torch.load(config.weights_root / "weights/HybridGNet/best.pt"))
hybrid.eval()

# =========================================================
config.layer = 3

Skip3 = Skip(config, D_t, U_t, A_t).to(config.device)
Skip3.load_state_dict(torch.load(config.weights_root / "weights/Skip/skip_L3/best.pt"))
Skip3.eval()

# =========================================================
config.layer = 4

Skip4 = Skip(config, D_t, U_t, A_t).to(config.device)
Skip4.load_state_dict(torch.load(config.weights_root / "weights/Skip/skip_L4/best.pt"))
Skip4.eval()

# =========================================================
config.layer = 5

Skip5 = Skip(config, D_t, U_t, A_t).to(config.device)
Skip5.load_state_dict(torch.load(config.weights_root / "weights/Skip/skip_L5/best.pt"))
Skip5.eval()

# =========================================================
config.layer = 6

Skip6 = Skip(config, D_t, U_t, A_t).to(config.device)
Skip6.load_state_dict(torch.load(config.weights_root / "weights/Skip/skip_L6/best.pt"))
Skip6.eval()

pcaNet = PCA_Net(config).to(config.device)
pcaNet.load_state_dict(torch.load(config.weights_root / 'weights/baselines/pca/best.pt'))
pcaNet.eval()

vae = VAE_Mixed(config).to(config.device)
vae.load_state_dict(torch.load(config.weights_root / 'weights/baselines/vae/best.pt'))
vae.eval()

modelUNet = UNet(n_classes = 3).to(config.device)
modelUNet.load_state_dict(torch.load(config.weights_root / 'weights/UNet/best.pt'))
modelUNet.eval()


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

model_list = [pcaNet, vae, hybrid, Skip3, Skip4, Skip5, Skip6, double43, double54, double65]
model_names = ['PCA', 'FC', 'HybridGNet', '1-SC Layer 3', '1-SC Layer 4','1-SC Layer 5','1-SC Layer 6', '2-SC Layers 4-3', '2-SC Layers 5-4', '2-SC Layers 6-5']

results1 = pd.DataFrame()

for i in range(0, len(test_dataset.images)):  
    print('\r',i+1,'of', len(test_dataset.images),end='')
    with torch.no_grad():
        sample = test_dataset[i]

        data, target = sample['image'], sample['landmarks']
        data = torch.unsqueeze(data, 0).to(config.device)
        target = target[:120,:].reshape(-1).numpy()
       
        for j in range(0, len(model_list)):
            print(model_names[j])
            output = model_list[j](data)
            if len(output) > 1:
                output = output[0]
            output = output.cpu().numpy().reshape(-1)
           
            error = mean_squared_error(target * 1024, output * 1024)
           
            aux = pd.DataFrame([[i, error, model_names[j]]], columns=['i','MSE', 'Model'])
            results1 = pd.concat([results1, aux], ignore_index=True)


import pathlib
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

folder = "MultiAtlas/JSRT/images/output_points"
test = "../Datasets/JSRT/Test/landmarks"

data_root = pathlib.Path(folder)
all_files = list(data_root.glob('*.npy'))
all_files = [str(path) for path in all_files]
all_files.sort(key = natural_key)

for i in range(0, len(all_files)):
    data = np.load(all_files[i])[:240]
    target = np.load(all_files[i].replace(folder, test))[:240]
    
    error = mean_squared_error(target, data)

    aux = pd.DataFrame([[i, error, "MultiAtlas"]], columns=['i','MSE', 'Model'])
    results1 = results1.append(aux, ignore_index = True)
    
model_names.append('MultiAtlas')


plt.figure(figsize = (10,5))
plt.tight_layout()
sns.boxplot(x = 'Model', y = 'MSE', data = results1, showmeans = True)
plt.xticks(rotation=25, ha="right" )
plt.ylabel('MSE')
plt.title('MSE')
plt.xlabel(None)

print('MSE')
for model in model_names:
    print(model, '\t' '%.3f'%np.mean(results1['MSE'][results1['Model'] == model]), '+- %.3f' % np.std(results1['MSE'][results1['Model'] == model]))

print('')


model_names.append('UNet')


aux = results1[results1['Model'] == '2-SC Layers 6-5']
aux0 = aux.sort_values(by = 'MSE')[-3:]
aux0


from utils.fun import drawOrgans

model_list_ = [pcaNet, vae, hybrid, Skip6, double65]
model_names_ = ['PCA', 'FC', 'HybridGNet', '1-IGSC Layer 6', '2-IGSC Layers 6-5']

i_ =[4, 13, 44]

fig = plt.figure(figsize=(24, 8), dpi= 200)

c = 0

for i in i_:
    with torch.no_grad():
        sample = test_dataset[i]

        data, target = sample['image'], sample['landmarks']
        data = torch.unsqueeze(data, 0).to(config.device)
        target = target.reshape(-1).numpy()

        draw = data.cpu().numpy()[0,0,:,:]
        
        ax = plt.subplot(3, len(model_list_) + 3, 1 + c * (len(model_list_) + 3))
        plt.axis('off')
        plt.xlim(1, 1024)
        plt.ylim(1024, 1)
        
        target = np.clip(target, 0, 1)
        if c == 0:
            drawOrgans(ax, target[:240] * 1024, None, draw.copy())
            plt.title("Ground Truth", fontsize = 20)
        else:
            drawOrgans(ax, target[:240] * 1024, None, draw.copy())
        
        ax = plt.subplot(3, len(model_list_) + 3, 2 + c * (len(model_list_) + 3))
        
        data_MA = test_dataset.images[i].replace('Datasets/JSRT/Test/Images', "Results/MultiAtlas/JSRT/images/output_points").replace(".png", ".npy")
        data_MA = np.load(data_MA)[:240]
        
        plt.axis('off')
        plt.xlim(1, 1024)
        plt.ylim(1024, 1)
        drawOrgans(ax, data_MA, None, draw.copy())
        if c == 0:
            plt.title("MultiAtlas", fontsize = 20)
        
        for j in range(0, len(model_list_)):
            output = model_list_[j](data)
            if len(output) > 1:
                output = output[0]
            output = output.cpu().numpy().reshape(-1) 
            output = np.clip(output, 0, 1)[:240]
            ax = plt.subplot(3, len(model_list_) + 3, j + 3 + c * (len(model_list_) + 3))
            plt.axis('off')
            if c == 0:
                drawOrgans(ax, output * 1024, None, draw.copy())
                plt.title(model_names_[j], fontsize = 20)
            else:
                drawOrgans(ax, output * 1024, None, draw.copy())
            
            plt.xlim(1, 1024)
            plt.ylim(1024, 1)

        
        ax = plt.subplot(3, len(model_list_) + 3, j + 4 + c * (len(model_list_) + 3))
        plt.axis('off')

        output = modelUNet(data)
        output = torch.argmax(output[0,:,:,:], axis=0).cpu().numpy()
        
        image=np.zeros(list(draw.shape) + [3])
        image[:,:,0] = draw + 0.7 * (output == 1).astype('float') - 0.3 * (output == 2).astype('float')
        image[:,:,1] = draw + 0.7 * (output == 2).astype('float') - 0.2 * (output == 1).astype('float')
        image[:,:,2] = draw - 0.2 * (output == 1).astype('float') - 0.3 * (output == 2).astype('float')
        image = np.clip(image, 0, 1)
        if c == 0:
            plt.title('UNet', fontsize = 20)
        plt.imshow(image)
        plt.xlim(1, 1024)
        plt.ylim(1024, 1)
        
        c += 1
        
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0)      
plt.savefig('figs/compare_UNet.png', bbox_inches = 'tight', dpi=200)
plt.savefig('figs/compare_UNet.pdf', bbox_inches = 'tight', dpi=200)
