import torch
import pandas as pd

from typing import List
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, test_dataset: Dataset, device: str):
        self.test_dataset = test_dataset
        self.device = device
        
    def evaluate_models(self, models: List[torch.nn.Module], model_names: List[str]) -> pd.DataFrame:
        results = []
        
        for i in range(len(self.test_dataset)):
            print(f'\r{i+1} of {len(self.test_dataset)}', end='')
            sample = self.test_dataset[i]
            data = torch.unsqueeze(sample['image'], 0).to(self.device)
            target = sample['landmarks'][:120,:].reshape(-1).numpy()
            
            for model, model_name in zip(models, model_names):
                print("111111111", model_name)
                with torch.no_grad():
                    output = model(data)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.cpu().numpy().reshape(-1)
                    
                    error = mean_squared_error(target * 1024, output * 1024)
                    results.append({
                        'i': i,
                        'MSE': error,
                        'Model': model_name
                    })
                    
        return pd.DataFrame(results)