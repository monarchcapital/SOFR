import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- 1. Configuration and Data Loading ---
SOFR_RATES_FILE = 'sofr rates.csv'

def run_sofr_pca():
    print(f"Loading data from '{SOFR_RATES_FILE}'...")
    try:
        # Load the data, using 'Date' column as the index
        df_sofr = pd.read_csv(SOFR_RATES_FILE, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: The file '{SOFR_RATES_FILE}' was not found. Please check the file name.")
        return

    # --- 2. Data Cleaning and Preparation ---

    # Drop columns/rows with no data and fill remaining NaNs using forward/backward fill
    df_sofr.dropna(axis=1, how='all', inplace=True)
    df_sofr.fillna(method='ffill', inplace=True)
    df_sofr.fillna(method='bfill', inplace=True)
    df_sofr.dropna(axis=0, how='any', inplace=True)

    if df_sofr.empty:
        print("Error: DataFrame is empty after cleaning. Cannot perform PCA.")
        return

    # Calculate Daily Changes (Returns), which is the standard input for risk factor PCA
    df_changes = df_sofr.diff().dropna()
    X = df_changes.values
    maturity_labels = df_sofr.columns.tolist()

    print(f"Analysis performed on {len(df_changes)} daily change observations.")
    print(f"Contracts analyzed: {', '.join(maturity_labels)}")

    # --- 3. Standardization and PCA Execution ---

    # Standardize data to ensure each contract contributes equally based on volatility
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA, retaining all components
    pca = PCA(n_components=None)
    pca.fit(X_scaled)

    # --- 4. Results Analysis ---

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = pca.components_

    print("\n" + "="*80)
    print("      PCA Results: Fixed Income Risk Factors (Level, Slope, Curvature)")
    print("="*80)

    # Summary of variance explained by the top 3 factors
    print("\n--- Explained Variance Ratio ---")
    for i in range(3):
        factor_name = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"][i]
        print(f"{factor_name:<15}: {explained_variance_ratio[i]*100:.2f}%")
    
    print(f"\nTotal Variance Explained by Top 3 Components: {cumulative_variance[2]*100:.2f}%")


    # Display Principal Component Loadings (Eigenvectors)
    df_loadings = pd.DataFrame(
        components,
        columns=maturity_labels,
        index=[f"PC{i+1}" for i in range(len(components))]
    )
    
    print("\n--- Principal Component Loadings (Top 3) ---")
    # Transpose for better readability (maturities as rows, PCs as columns)
    df_display = df_loadings.iloc[:3].T 
    
    # Add a column for the time-to-maturity of each contract, if possible (requires EXPIRY.csv)
    # Since we don't have a clean way to integrate EXPIRY.csv here, we stick to contract labels.

    print(df_display.to_markdown(floatfmt=".4f"))
    
    # --- 5. Visualization: Cumulative Explained Variance ---

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='-', color='#1e40af', linewidth=2)
    
    plt.title('Cumulative Explained Variance (Scree Plot)', fontsize=16)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    
    # Highlight the 95% threshold
    plt.axhline(y=0.95, color='#b91c1c', linestyle='--', label='95% Threshold')
    plt.xticks(np.arange(1, len(cumulative_variance) + 1, 2))
    plt.grid(axis='y', linestyle='dotted', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Encode plot to base64 for display in the environment
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(plt.gcf()) 
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    
    # This output tag instructs the interactive environment to render the image
    print(f"\n\n\n{plot_data}\n[End Image Data]")


if __name__ == "__main__":
    run_sofr_pca()
