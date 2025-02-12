import torch
import torch.nn as nn
from typing import Type

from models.hybrid import Hybrid
from config import ModelConfig
from data_utils import load_sparse_matrices


class ModelLoader:
    def __init__(self, config: ModelConfig, device: str):
        self.config = config
        self.device = device
        self.A_t, self.D_t, self.U_t = load_sparse_matrices(device)
        
    def load_model(self, model_class: Type[nn.Module], weights_path: str, **kwargs) -> nn.Module:

        if issubclass(model_class, Hybrid):
            model = model_class(self.config, self.D_t, self.U_t, self.A_t, **kwargs).to(self.device)
        else:
            model = model_class(self.config, **kwargs).to(self.device)

        model.load_state_dict(torch.load(weights_path))
        model.eval()
        return model