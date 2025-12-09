import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid", context="paper")

def load_data():
    full_data = pd.read_parquet('data/processed/full_data.parquet')
    validation = pd.read_parquet('data/processed/validation3990_data.parquet')
    try:
        results = pd.read_csv('results/modeling_results.csv')
    except FileNotFoundError:
        results = None
    return full_data, validation, results

def fig_s1a(full_data):
    """
    Scatter plot of RT vs m/z, colored by excluded/included (proxied by clean data presence).
    Since full_data contains excluded rows if we saved them? 
    Wait, process_data filtered them out.
    If we want to show excluded, we need the raw data again.
    For now, we plot what we have.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot
    sns.scatterplot(data=full_data, x='row_retention_time', y='row_m_z', 
                    hue='non_heavy_identified_flag', style='non_heavy_identified_flag',
                    palette={0: 'grey', 1: 'black'}, alpha=0.6, s=15)
    
    plt.xlabel("Retention time (mins)")
    plt.ylabel("Mass/Charge ratio")
    plt.title("Figure S1A: Discovery Data")
    plt.xlim(0, 16)
    plt.ylim(50, 900)
    plt.tight_layout()
    plt.savefig('figures/Figure_S1A.png')
    plt.close()

def fig_1b(full_data, validation):
    """
    Metabolites (%) by charge (Pos/Neg).
    """
    # Prepare data
    d1 = full_data.copy()
    d1['Dataset'] = 'Discovery'
    d2 = validation.copy()
    d2['Dataset'] = 'Validation'
    
    # Needs charge column. utils/process_data created it or we split row_id.
    # We should have saved charge in process_data. If not, split again.
    for d in [d1, d2]:
        if 'charge' not in d.columns:
             d[['charge', 'id_temp']] = d['row_id'].str.split('_', n=1, expand=True)

    combined = pd.concat([d1, d2], ignore_index=True)
    
    # Percentage
    counts = combined.groupby(['Dataset', 'charge']).size().reset_index(name='count')
    totals = combined.groupby(['Dataset']).size().reset_index(name='total')
    counts = pd.merge(counts, totals, on='Dataset')
    counts['perc'] = counts['count'] / counts['total'] * 100
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=counts, x='Dataset', y='perc', hue='charge', palette='viridis')
    plt.ylabel("Metabolites (%)")
    plt.title("Figure 1B: Metabolite Charge Content")
    plt.tight_layout()
    plt.savefig('figures/Figure_1B.png')
    plt.close()

def fig_s2_performance(results):
    if results is None:
        print("No modeling results found. Skipping metrics plots.")
        return
        
    # Rank models by accuracy
    # R plot: x=Rank, y=Accuracy, Shape=Recipe, Color=Model
    
    # Taking max mean accuracy per workflow (already done in loop? we saved all results)
    # We simplified so we have 1 row per model+preprocess.
    
    results = results.sort_values(by='test_accuracy', ascending=False).reset_index(drop=True)
    results['Rank'] = results.index + 1
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results, x='Rank', y='test_accuracy', 
                    style='preprocess', hue='model', s=100)
    
    plt.ylim(0.5, 1.0)
    plt.title("Model Performance (Accuracy)")
    plt.tight_layout()
    plt.savefig('figures/Figure_S2_Accuracy.png')
    plt.close()
    
    # ROC AUC
    results = results.sort_values(by='test_roc_auc', ascending=False).reset_index(drop=True)
    results['Rank'] = results.index + 1
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results, x='Rank', y='test_roc_auc', 
                    style='preprocess', hue='model', s=100)
    
    plt.ylim(0.5, 1.0)
    plt.title("Model Performance (ROC AUC)")
    plt.tight_layout()
    plt.savefig('figures/Figure_ROC_AUC.png')
    plt.close()

def main():
    os.makedirs('figures', exist_ok=True)
    full_data, validation, results = load_data()
    
    print("Generating Figure S1A...")
    fig_s1a(full_data)
    
    print("Generating Figure 1B...")
    fig_1b(full_data, validation)
    
    print("Generating Figure S2...")
    fig_s2_performance(results)
    
    print("Figures generated in figures/")

if __name__ == "__main__":
    main()
