import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np
from kindel.utils.data import rmse, spearman
from scipy import stats
import pandas as pd


def plot_regression_metrics(y_true_on, y_pred_on, y_true_off, y_pred_off, 
                          title="Prediction", output_dir="results/plots", 
                          log_wandb=False, dataset_type="extended",
                          train_metric=None, valid_metric=None, test_metric=None,
                          metric_name="MSE"):
    """Plot regression metrics with predicted enrichment vs 1/experimental_kd."""
    
    # Convert inputs to numpy arrays
    y_true_on, y_pred_on = np.array(y_true_on), np.array(y_pred_on)
    y_true_off, y_pred_off = np.array(y_true_off), np.array(y_pred_off)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert Kd to 1/Kd (higher values mean stronger binding)
    y_true_on_inv = 1/np.log10(y_true_on)
    y_true_off_inv = 1/np.log10(y_true_off)
    
    # Calculate metrics
    rho_on = spearman(y_pred_on, y_true_on_inv)
    pearson_on = stats.pearsonr(y_pred_on, y_true_on_inv)[0]
    rho_off = spearman(y_pred_off, y_true_off_inv)
    pearson_off = stats.pearsonr(y_pred_off, y_true_off_inv)[0]
    
    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # On-DNA plot
    sns.scatterplot(x=y_true_on_inv, y=y_pred_on, alpha=0.5, ax=ax1)
    metrics_text_on = (f'Train {metric_name}: {train_metric:.3f}\n'
                      f'Valid {metric_name}: {valid_metric:.3f}\n'
                      f'Test {metric_name}: {test_metric:.3f}\n'
                      f'Spearman ρ: {rho_on:.3f}\n'
                      f'Pearson r: {pearson_on:.3f}')
    ax1.text(0.05, 0.95, metrics_text_on,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    ax1.set_xlabel('1/Experimental Kd')
    ax1.set_ylabel('Predicted Enrichment')
    ax1.set_title('On-DNA Predictions')
    
    # Off-DNA plot
    sns.scatterplot(x=y_true_off_inv, y=y_pred_off, alpha=0.5, ax=ax2)
    metrics_text_off = (f'Spearman ρ: {rho_off:.3f}\n'
                       f'Pearson r: {pearson_off:.3f}')
    ax2.text(0.05, 0.95, metrics_text_off,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    ax2.set_xlabel('1/Experimental Kd')
    ax2.set_ylabel('Predicted Enrichment')
    ax2.set_title('Off-DNA Predictions')
    
    plt.suptitle(f"{title} ({dataset_type})")
    
    # Save plot
    base_path = os.path.join(output_dir, f"{title}_{dataset_type}".replace(' ', '_'))
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    
    # Save data to separate CSVs
    df_on = pd.DataFrame({
        'experimental_kd_inverse': y_true_on_inv,
        'predicted_enrichment': y_pred_on,
    })
    df_on.to_csv(f"{base_path}_on_DNA.csv", index=False)
    
    df_off = pd.DataFrame({
        'experimental_kd_inverse': y_true_off_inv,
        'predicted_enrichment': y_pred_off,
    })
    df_off.to_csv(f"{base_path}_off_DNA.csv", index=False)
    
    if log_wandb:
        wandb.log({f"regression_plot_{title}_{dataset_type}": wandb.Image(fig)})
    
    plt.close(fig)
    return fig
