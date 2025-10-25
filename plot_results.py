# FILE: plot_results.py

import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Load Data from CSV Files ---

results_dir = 'results'
data_files = {
    "FedAvg (alpha=1.0)": "fedavg_baseline.csv",
    "Fed-PEKD RealSamples (alpha=1.0)": "fedpekd_agg_real-samples.csv", # Make sure this file exists from your baseline run
    "Fed-PEKD Prototypes (Collapsed, alpha=1.0)": "fedpekd_prototypes_v1.csv",
    "Fed-PEKD ProtoGMM (alpha=1.0)": "fedpekd_protogmm_v2_robust.csv", # The one that decayed
    "Fed-PEKD ProtoGMM (alpha=5.0 - Best)": "fedpekd_protogmm_v3_alpha5.csv" # The final stable run
}

plt.figure(figsize=(14, 9))

for label, filename in data_files.items():
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Use different linestyles/markers for clarity
        linestyle = '--' if 'FedAvg' in label else '-'
        marker = 'x' if 'FedAvg' in label else ('o' if 'Best' in label else '.')
        # Make the "Best" line thicker
        linewidth = 3 if 'Best' in label else 1.5
        
        plt.plot(df['round'], df['accuracy'], marker=marker, linestyle=linestyle, linewidth=linewidth, label=label)
    else:
        print(f"Warning: Results file not found, skipping plot: {filepath}")

# --- Formatting ---
plt.title('Fed-PEKD Variants vs. FedAvg on Non-IID FashionMNIST', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Global Model Accuracy (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.ylim(0, 100)
plt.xlim(0, 50) # Assuming max 50 rounds

# --- Save the Plot ---
plt.savefig('results/ALL_VARIANTS_Comparison.png')
print("Comparison plot saved as 'results/ALL_VARIANTS_Comparison.png'")
plt.show()