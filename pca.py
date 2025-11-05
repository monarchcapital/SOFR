import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime

# --- Configuration ---
SOFR_RATES_FILE = 'sofr rates.csv'
EXPIRY_FILE = 'EXPIRY (2).csv'

# --- Helper Functions ---

def load_data():
    """Loads required CSV data (SOFR rates and expiry dates)."""
    print(f"Loading data from '{SOFR_RATES_FILE}' and '{EXPIRY_FILE}'...")
    try:
        # Load Price File
        df_prices = pd.read_csv(
            SOFR_RATES_FILE,
            index_col='Date',
            parse_dates=True
        )
        # Load Expiry File
        df_expiry = pd.read_csv(EXPIRY_FILE)
        df_expiry = df_expiry.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
        df_expiry = df_expiry.set_index('Contract')
        df_expiry['ExpiryDate'] = pd.to_datetime(df_expiry['ExpiryDate'])

    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}. Please ensure both files are available.")
        return None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None

    # Cleaning and Preprocessing for Prices
    df_prices.dropna(axis=1, how='all', inplace=True)
    df_prices.fillna(method='ffill', inplace=True)
    df_prices.fillna(method='bfill', inplace=True)
    df_prices.dropna(axis=0, how='any', inplace=True)

    if df_prices.empty:
        print("Error: Price DataFrame is empty after cleaning. Cannot proceed.")
        return None, None

    return df_prices, df_expiry

def calculate_pca(df_prices, n_components=5):
    """Calculates PCA on daily changes (returns) of SOFR outright contracts."""

    # Calculate Daily Changes (Returns)
    df_changes = df_prices.diff().dropna()
    X = df_changes.values
    maturity_labels = df_prices.columns.tolist()

    if X.shape[0] < 2:
        print("Error: Not enough data points to calculate PCA after differencing.")
        return None, None, None, None, None

    # Standardize data (crucial for volatility-based PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)


    # Prepare results
    df_loadings = pd.DataFrame(
        pca.components_,
        columns=maturity_labels,
        index=[f"PC{i+1}" for i in range(n_components)]
    )

    df_factors = pd.DataFrame(
        X_pca,
        index=df_changes.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    explained_variance = pca.explained_variance_ratio_

    # Return transposed loadings for better table printing
    return df_loadings.T, df_factors, explained_variance, scaler, pca


def prepare_butterflies(df_prices, df_expiry):
    """Creates butterfly spreads (3 contracts) and their time-to-maturity."""
    contracts = df_prices.columns.tolist()
    butterflies = {}

    # Generate all 3-contract flies (k-1, k, k+1)
    for i in range(1, len(contracts) - 1):
        front = contracts[i-1]
        mid = contracts[i]
        back = contracts[i+1]

        # Check if all contracts exist in the expiry data for maturity calculations
        if front in df_expiry.index and mid in df_expiry.index and back in df_expiry.index:
            # Fly calculation: Front - 2*Mid + Back
            fly_prices = df_prices[front] - 2 * df_prices[mid] + df_prices[back]
            fly_name = f"{front}-{mid}-{back}"
            butterflies[fly_name] = fly_prices.dropna()

    if not butterflies:
        print("Warning: No valid butterfly spreads could be constructed.")
        return None, None

    df_butterflies = pd.DataFrame(butterflies)

    # Calculate time-to-maturity for the mid-leg of each fly
    maturity_data = {}
    for fly_name, fly_series in df_butterflies.items():
        mid_contract = fly_name.split('-')[1]

        # Only proceed if we have expiry data
        if mid_contract in df_expiry.index:
            expiry_date = df_expiry.loc[mid_contract, 'ExpiryDate']

            # Use the date index of the fly series to calculate days to maturity
            days_to_maturity = (expiry_date - fly_series.index).days / 365.25
            maturity_data[fly_name] = days_to_maturity

    df_maturity = pd.DataFrame(maturity_data)

    return df_butterflies, df_maturity

def calculate_mispricing(df_prices, df_loadings, df_factors, scaler, pca, analysis_date):
    """
    Calculates the 'fair' butterfly price based on the top two principal components
    and determines the mispricing on a specific analysis date.
    """

    if analysis_date not in df_factors.index:
        print(f"Error: Analysis date {analysis_date.strftime('%Y-%m-%d')} not found in PCA factors.")
        return None, None, None

    # 1. Get Outright Prices (Original and PCA Fair)

    # The actual outright price changes on the analysis date
    outright_changes = df_prices.diff().loc[analysis_date].values.reshape(1, -1)

    # Standardize the changes using the original scaler
    outright_changes_scaled = scaler.transform(outright_changes)

    # Project actual changes onto all PCs
    actual_factors = pca.transform(outright_changes_scaled)

    # Reconstitute using ONLY the top 2 PCs (PC1/Level and PC2/Slope)
    # Step 1: Zero out all factors beyond PC2
    factors_2pc = actual_factors.copy()
    factors_2pc[:, 2:] = 0 # Zero out PC3, PC4, ...

    # Step 2: Inverse transform to get the 2PC-only imputed *changes*
    imputed_changes_scaled = pca.inverse_transform(factors_2pc)

    # Step 3: Inverse the scaling to get the 2PC-only imputed *price changes*
    imputed_changes_2pc = scaler.inverse_transform(imputed_changes_scaled)

    # Get the original outright prices (LEVEL) on the analysis date
    original_outright_prices = df_prices.loc[analysis_date].copy()

    # Find the outright prices for the day BEFORE the analysis date to reconstruct the level
    prev_date_idx = df_prices.index.get_loc(analysis_date) - 1
    if prev_date_idx < 0:
        print("Error: No previous day data available for reconstruction.")
        return None, None, None

    prev_date_prices = df_prices.iloc[prev_date_idx]

    # Reconstituted outright prices (LEVEL) based on 2PC-only changes
    # Reconstituted Price = Prev Day Price + 2PC-Imputed Change
    reconstituted_outright_prices_2pc = prev_date_prices + imputed_changes_2pc[0]

    # 2. Calculate Original and PCA Fair Butterfly Prices

    butterfly_comparison = {}

    # Get the contract list
    contracts = df_prices.columns.tolist()

    for i in range(1, len(contracts) - 1):
        front = contracts[i-1]
        mid = contracts[i]
        back = contracts[i+1]
        fly_name = f"{front}-{mid}-{back}"

        # Original Fly Price = Front - 2*Mid + Back (using original prices)
        original_fly = (original_outright_prices[front] -
                        2 * original_outright_prices[mid] +
                        original_outright_prices[back])

        # PCA Fair Fly Price = Front - 2*Mid + Back (using 2PC-only prices)
        pca_fair_fly = (reconstituted_outright_prices_2pc[front] -
                        2 * reconstituted_outright_prices_2pc[mid] +
                        reconstituted_outright_prices_2pc[back])

        # Mispricing is the difference
        mispricing = original_fly - pca_fair_fly

        butterfly_comparison[fly_name] = {
            'Original': original_fly,
            'PCA Fair': pca_fair_fly,
            'Mispricing': mispricing
        }

    df_comparison = pd.DataFrame.from_dict(butterfly_comparison, orient='index')

    return df_comparison, original_outright_prices, reconstituted_outright_prices_2pc

def plot_mispricing(df_comparison, df_maturity, analysis_date):
    """Generates a base64 image of the butterfly mispricing vs time-to-maturity."""

    # Ensure the maturity data aligns with the comparison data's index
    valid_flies = df_comparison.index.intersection(df_maturity.columns)

    if valid_flies.empty:
        print("Warning: Cannot plot mispricing. No overlapping contracts between comparison and maturity data.")
        return None

    df_plot = df_comparison.loc[valid_flies].copy()

    # Get Time-to-Maturity (in years) for the mid-leg of the fly on the analysis date
    time_to_maturity = df_maturity.loc[analysis_date, valid_flies]

    # Convert mispricing to BPS (Basis Points)
    df_plot['Mispricing (BPS)'] = df_plot['Mispricing'] * 10000
    df_plot['Time to Maturity (Years)'] = time_to_maturity

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Plotting Mispricing vs Time-to-Maturity
    # Use a diverging color map to show positive/negative mispricing
    plt.scatter(
        df_plot['Time to Maturity (Years)'],
        df_plot['Mispricing (BPS)'],
        c=df_plot['Mispricing (BPS)'],
        cmap='coolwarm',
        s=150, # size of points
        edgecolors='k',
        alpha=0.7
    )

    # Label the points with the mid-contract name
    for i, row in df_plot.iterrows():
        mid_contract = i.split('-')[1]
        plt.annotate(
            mid_contract,
            (row['Time to Maturity (Years)'] + 0.05, row['Mispricing (BPS)']),
            fontsize=9
        )

    plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)

    plt.title(f'Butterfly Mispricing vs. Time-to-Maturity ({analysis_date.strftime("%Y-%m-%d")})',
              fontsize=16)
    plt.xlabel('Time to Mid-Contract Maturity (Years)', fontsize=14)
    plt.ylabel('Mispricing (Basis Points)', fontsize=14)
    plt.colorbar(label='Mispricing (BPS)')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Encode plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(plt.gcf())
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    return plot_data

# Main function to execute the console analysis
def run_full_pca_analysis():
    print("="*80)
    print("SOFR Futures PCA & Butterfly Mispricing Analysis (Console Output)")
    print("="*80)

    df_prices, df_expiry = load_data()

    if df_prices is None or df_expiry is None:
        return

    # Use the latest available date for analysis
    analysis_date = df_prices.index[-1]
    print(f"\nAnalysis Date automatically set to the latest data point: {analysis_date.strftime('%Y-%m-%d')}")

    # --- 1. Outright PCA Calculation ---
    n_components = min(df_prices.shape[1], 5) # Use max 5 components or fewer if less contracts
    print(f"\n--- 1. Performing PCA on Outright Futures Changes using {n_components} components ---")
    df_loadings, df_factors, explained_variance, scaler, pca = calculate_pca(df_prices, n_components)

    if df_loadings is None:
        print("PCA calculation failed. Exiting.")
        return

    # Print Explained Variance
    print("\n##### Explained Variance Ratio #####")
    cumulative_variance = np.cumsum(explained_variance)
    for i in range(len(explained_variance)):
        pc_name = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"][i] if i < 3 else f"PC{i+1}"
        print(f"{pc_name:<15}: {explained_variance[i]*100:.2f}% (Cumulative: {cumulative_variance[i]*100:.2f}%)")

    # Plot Explained Variance (Scree Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='-', color='#1e40af', linewidth=2)
    plt.title('Cumulative Explained Variance (Scree Plot)', fontsize=16)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    plt.axhline(y=0.95, color='#b91c1c', linestyle='--', label='95% Threshold')
    plt.xticks(np.arange(1, len(cumulative_variance) + 1, 1))
    plt.grid(axis='y', linestyle='dotted', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(plt.gcf())
    plot_data_scree = base64.b64encode(buffer.getvalue()).decode()
    print(f"\n\n\n{plot_data_scree}\n[End Image Data]")

    # Print PC Loadings
    print("\n##### Principal Component Loadings (Top 3 - Transposed) #####")
    print("These are the weights (eigenvectors) of each contract on the first three risk factors.")
    df_display_loadings = df_loadings.iloc[:, :3] # Select top 3 PCs
    df_display_loadings.index.name = 'Contract'
    print(df_display_loadings.to_markdown(floatfmt=".4f"))


    # --- 2. Butterfly Spread Mispricing Calculation ---
    print("\n--- 2. Calculating Butterfly Spreads and Mispricing ---")

    df_butterflies, df_maturity = prepare_butterflies(df_prices, df_expiry)

    if df_butterflies is None:
        return

    # Calculate Mispricing
    df_comparison, _, _ = calculate_mispricing(
        df_prices, df_loadings, df_factors, scaler, pca, analysis_date
    )

    if df_comparison is None:
        print("Mispricing calculation failed. Exiting.")
        return

    print("\n##### Butterfly Mispricing (Original vs. 2PC Fair Price) #####")
    print("Mispricing is calculated as Original Fly - PCA Fair Fly (2-factor model).")

    # Prepare detailed comparison table
    detailed_comparison_fly = df_comparison.copy()
    detailed_comparison_fly.index.name = 'Butterfly Contract'
    detailed_comparison_fly['Mispricing (BPS)'] = detailed_comparison_fly['Mispricing'] * 10000
    detailed_comparison_fly = detailed_comparison_fly.rename(
        columns={'Original': 'Original Fly', 'PCA Fair': 'PCA Fair Fly'}
    )

    # Add mid-contract maturity in years for context
    time_to_maturity_series = df_maturity.loc[analysis_date, detailed_comparison_fly.index].rename('Time to Maturity (Years)')
    detailed_comparison_fly = detailed_comparison_fly.join(time_to_maturity_series)

    # Reorder columns and format output
    output_columns = ['Original Fly', 'PCA Fair Fly', 'Mispricing (BPS)', 'Time to Maturity (Years)']
    print(detailed_comparison_fly[output_columns].to_markdown(floatfmt=".4f"))


    # Plot Mispricing vs Time-to-Maturity
    plot_data_mispricing = plot_mispricing(df_comparison, df_maturity, analysis_date)

    if plot_data_mispricing:
        print(f"\n\n\n{plot_data_mispricing}\n[End Image Data]")


if __name__ == "__main__":
    # The main execution block
    run_full_pca_analysis()
