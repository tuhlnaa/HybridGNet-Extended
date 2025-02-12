
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict

def plot_model_comparison(results: pd.DataFrame, model_names: List[str], save_path: str = None):
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    sns.boxplot(x='Model', y='MSE', data=results, showmeans=True)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel('MSE')
    plt.title('Model Performance Comparison')
    plt.xlabel(None)
    
    print('MSE Results:')
    for model in model_names:
        model_results = results[results['Model'] == model]
        print(f"{model:<20} {model_results['MSE'].mean():.3f} ± {model_results['MSE'].std():.3f}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()