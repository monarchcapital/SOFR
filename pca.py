import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer")

# --- Helper Functions for Data Processing ---

def load_data(uploaded_file):
    """Loads CSV data into a DataFrame."""
    if uploaded_file is not None:
        try:
            # Assuming dates are in the first column for prices
            if 'price' in uploaded_file.name.lower():
                # Use thousands=',' option if prices might have commas (common in finance CSVs)
                df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                df.index.name = 'Date'
                return df
            # Assuming expiry data is simpler: Contract, ExpiryDate
            elif 'expiry' in uploaded_file.name.lower():
                df = pd.read_csv(uploaded_file)
                df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
                df = df.set_index('Contract')
                return df
            else:
                st.error("Please ensure file names clearly indicate 'price' or 'expiry'.")
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def get_generic_maturity_map(expiry_df, analysis_date):
    """
    Maps contract codes (e.g., U5) to generic quarterly maturities (Q1, Q2, Q3, ...)
    based on the nearest expiry date *after* the analysis date.
    """
    if expiry_df is None:
        return {}
    
    # Filter contracts that expire on or after the analysis date
    # The first contract (Q1) is the one that expires soonest after the analysis date.
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
        return {}

    # Sort by expiry date to determine Q1, Q2, etc.
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    # Create generic maturity labels (Q1, Q2, Q3, ...)
    future_expiries['GenericMaturity'] = [f'Q{i+1}' for i in range(len(future_expiries))]
    
    return future_expiries['GenericMaturity'].to_dict()

def transform_to_generic_curve(price_df, maturity_map):
    """
    Transforms the price DataFrame from contract codes to generic maturity columns.
    Example: 'U5' column becomes 'Q1_Price' column based on the maturity map.
    """
    if price_df is None or not maturity_map:
        return pd.DataFrame()

    # Create the reverse map: Contract -> Generic_Label_Price
    reverse_map = {contract: label for contract, label in maturity_map.items()}

    # Filter price columns to only include those present in the reverse_map
    contract_cols = list(set(price_df.columns) & set(reverse_map.keys()))
    
    if not contract_cols:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame()

    # Rename columns using the generic maturity labels
    rename_dict = {contract: f"{reverse_map[contract]}" for contract in contract_cols}
    generic_df = price_df[contract_cols].rename(columns=rename_dict)
    
    # Ensure Q1, Q2, Q3... order for spreads
    sorted_cols = sorted(generic_df.columns, key=lambda x: int(x.strip('Q')))
    generic_df = generic_df[sorted_cols]
    
    return generic_df

def calculate_outright_spreads(generic_df):
    """
    Calculates the first differences (spreads) on a CME basis: C1 - C2, C2 - C3, etc.
    The spreads are labeled based on the shorter maturity contract (C1).
    """
    if generic_df.empty:
        return pd.DataFrame()

    num_contracts = generic_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - 1):
        # CME Basis: Shorter maturity minus longer maturity
        # Q1 - Q2, Q2 - Q3, ...
        short_maturity = generic_df.columns[i]
        long_maturity = generic_df.columns[i+1]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        
        spreads_data[spread_label] = generic_df.iloc[:, i] - generic_df.iloc[:, i+1]
        
    return pd.DataFrame(spreads_data)

def calculate_butterflies(generic_df):
    """
    Calculates butterflies (flies) on a CME basis: (Q1 - Q2) - (Q2 - Q3) = Q1 - 2*Q2 + Q3, etc.
    The flies are labeled based on the central contract (Q2).
    """
    if generic_df.empty or generic_df.shape[1] < 3:
        return pd.DataFrame()

    num_contracts = generic_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2):
        short_maturity = generic_df.columns[i]    # Q1
        center_maturity = generic_df.columns[i+1] # Q2
        long_maturity = generic_df.columns[i+2]   # Q3

        # Fly = Q1 - 2*Q2 + Q3
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = generic_df.iloc[:, i] - 2 * generic_df.iloc[:, i+1] + generic_df.iloc[:, i+2]

    return pd.DataFrame(flies_data)

def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads)."""
    if data_df.empty or data_df.shape[0] < data_df.shape[1]:
        st.error("Not enough data points (rows) to perform PCA on the spreads/flies.")
        return None, None, None

    # Standardize the data (PCA is sensitive to scale)
    data_scaled = (data_df - data_df.mean()) / data_df.std()
    data_scaled = data_scaled.fillna(0) # Fill NaNs that might result from constant columns

    # Determine optimal number of components (min(n_samples, n_features))
    n_components = min(data_scaled.shape)
    if n_components < 1:
        st.error("Cannot perform PCA with zero features.")
        return None, None, None

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Component Loadings (the eigenvectors * sqrt(eigenvalues))
    # We use the components array for factor loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    # Principal Component Scores (the transformed data)
    scores = pca.transform(data_scaled)
    
    return loadings, explained_variance, scores

def reconstruct_prices_and_derivatives(generic_df, reconstructed_spreads_df, spreads_df, butterflies_df):
    """
    Reconstructs Outright Prices and Butterflies historically using reconstructed spreads,
    anchored to the original Q1 price path (Level factor).
    """
    # --- 1. Reconstruct Outright Prices ---
    
    # Anchor the entire curve reconstruction to the original historical Q1 price path
    Q1_prices_original = generic_df.iloc[:, 0]
    
    # Initialize the reconstructed prices DataFrame, starting with the original Q1 as the Level anchor
    reconstructed_prices_df = pd.DataFrame(index=generic_df.index)
    reconstructed_prices_df[generic_df.columns[0] + ' (PCA)'] = Q1_prices_original
    
    # Iterate through all maturities starting from Q2 (index 1)
    for i in range(1, len(generic_df.columns)):
        prev_maturity = generic_df.columns[i-1]
        current_maturity = generic_df.columns[i]
        spread_label = f"{prev_maturity}-{current_maturity}"
        
        # Calculate the reconstructed price P_i using P_i-1 (PCA) and S_i-1,i (PCA)
        # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
        reconstructed_prices_df[current_maturity + ' (PCA)'] = (
            reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_df[spread_label]
        )
        
    # Merge original prices for comparison
    original_price_rename = {col: col + ' (Original)' for col in generic_df.columns}
    original_prices_df = generic_df.rename(columns=original_price_rename)
    
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Prepare Spreads for comparison ---
    original_spread_rename = {col: col + ' (Original)' for col in spreads_df.columns}
    pca_spread_rename = {col: col + ' (PCA)' for col in reconstructed_spreads_df.columns}

    original_spreads = spreads_df.rename(columns=original_spread_rename)
    pca_spreads = reconstructed_spreads_df.rename(columns=pca_spread_rename)
    
    historical_spreads = pd.merge(original_spreads, pca_spreads, left_index=True, right_index=True)
    
    
    # --- 3. Reconstruct Butterflies ---
    if butterflies_df.empty:
        return historical_outrights, historical_spreads, pd.DataFrame()
        
    reconstructed_butterflies = {}
    for i in range(len(spreads_df.columns) - 1):
        spread1_label = spreads_df.columns[i]
        spread2_label = spreads_df.columns[i+1]
        
        # Determine the fly label based on the original structure
        original_fly_label = butterflies_df.columns[i]
        
        # Reconstruct fly: Fly = Spread1_PCA - Spread2_PCA
        reconstructed_butterflies[original_fly_label + ' (PCA)'] = (
            reconstructed_spreads_df[spread1_label] - reconstructed_spreads_df[spread2_label]
        )

    reconstructed_butterflies_df = pd.DataFrame(reconstructed_butterflies, index=spreads_df.index)
    
    # Merge original flies for comparison
    original_fly_rename = {col: col + ' (Original)' for col in butterflies_df.columns}
    original_butterflies_df = butterflies_df.rename(columns=original_fly_rename)
    
    historical_butterflies = pd.merge(original_butterflies_df, reconstructed_butterflies_df, left_index=True, right_index=True)
    
    return historical_outrights, historical_spreads, historical_butterflies


# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (Date, U5, Z5, ...)", 
    type=['csv'], 
    key='price_upload'
)
expiry_file = st.sidebar.file_uploader(
    "Upload Contract Expiry Dates (Contract, ExpiryDate)", 
    type=['csv'], 
    key='expiry_upload'
)

# Initialize dataframes
price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

if price_df is not None:
    # --- Date Range Filter ---
    st.sidebar.header("2. Historical Date Range")
    min_date = price_df.index.min().date()
    max_date = price_df.index.max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Historical Data Range", 
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    price_df_filtered = price_df[(price_df.index.date >= start_date) & (price_df.index.date <= end_date)]
    
    # --- Analysis Date Selector (Maturity Roll) ---
    st.sidebar.header("3. Maturity Roll Date")
    
    # Analysis date should be within the historical range for stability
    default_analysis_date = end_date
    if default_analysis_date < min_date:
        default_analysis_date = min_date
        
    analysis_date = st.sidebar.date_input(
        "Select Analysis Date (Drives Q1, Q2, ... Mapping)", 
        value=default_analysis_date,
        min_value=min_date,
        max_value=max_date,
        key='analysis_date'
    )
    
    # Ensure analysis_date is a datetime object for comparison
    analysis_dt = datetime.combine(analysis_date, datetime.min.time())
    
else:
    st.info("Please upload both the Price Data and Expiry Data CSV files to begin the analysis.")
    st.stop()


# --- Core Processing Logic ---
if expiry_df is not None and not price_df_filtered.empty:
    
    # 1. Get the generic maturity map based on the analysis date
    maturity_map = get_generic_maturity_map(expiry_df, analysis_dt)
    
    if not maturity_map:
        st.warning("Could not establish generic maturity mapping. Please check if the 'Analysis Date' is before any contract expiry.")
        st.stop()
        
    # 2. Transform historical prices to generic curve prices
    generic_df = transform_to_generic_curve(price_df_filtered, maturity_map)

    if generic_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Spreads and Butterflies (Inputs for PCA and comparison)
    st.header("1. Data Derivatives Check")
    
    # Calculate Spreads
    spreads_df = calculate_outright_spreads(generic_df)
    st.markdown("##### Outright Spreads (Q1-Q2, Q2-Q3, etc.)")
    st.dataframe(spreads_df.head(5))
    
    if spreads_df.empty:
        st.warning("Spreads could not be calculated. Need at least two generic contracts (Q1, Q2).")
        st.stop()
        
    # Calculate Butterflies
    butterflies_df = calculate_butterflies(generic_df)
    st.markdown("##### Butterflies (Q1-2xQ2+Q3, etc.)")
    st.dataframe(butterflies_df.head(5))

    # 4. Perform PCA on Spreads
    # PCA is performed only on spreads, as they are the most stationary features
    loadings, explained_variance, scores = perform_pca(spreads_df)

    if loadings is not None:
        
        # --- Explained Variance Visualization ---
        st.header("2. Explained Variance")
        variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance (%)': explained_variance * 100
        })
        variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
        col_var, col_pca_select = st.columns([1, 1])
        with col_var:
            st.dataframe(variance_df, use_container_width=True)
            
        # Determine how many components to use for fair curve reconstruction
        default_pc_count = min(3, len(explained_variance))
        with col_pca_select:
            st.subheader("Fair Curve Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve:",
                min_value=1,
                max_value=len(explained_variance),
                value=default_pc_count,
                help="Typically, the first 3 components (Level, Slope, Curve) explain over 95% of variance in *spread changes*."
            )
            
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads.")
        
        
        # --- Component Loadings Heatmap ---
        st.header("3. Component Loadings (Heatmap)")
        st.markdown("Shows the weights of the first three PCs on each **Spread** (e.g., Q1-Q2, Q2-Q3). These represent the classic Level, Slope, and Curvature factors.")
        
        plt.style.use('default') # Use default style for consistency
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use only the top 3 (or fewer if not available) PCs for the heatmap visualization
        loadings_plot = loadings.iloc[:, :default_pc_count]

        sns.heatmap(
            loadings_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=0.5, 
            linecolor='gray', 
            cbar_kws={'label': 'Loading Weight'}
        )
        ax.set_title(f'Component Loadings for First {default_pc_count} Principal Components', fontsize=16)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Spread Contract (Generic Maturity)')
        st.pyplot(fig)
        
        # --- Historical Reconstruction ---
        
        # 1. Reconstruct Spreads using only selected PCs
        # Spreads_Reconstructed = (Scores @ Loadings.T) * Std + Mean
        
        # Scores used for reconstruction are only the first 'pc_count'
        scores_used = scores[:, :pc_count]
        loadings_used = loadings.iloc[:, :pc_count]
        
        # Reconstruct scaled data
        reconstructed_scaled = scores_used @ loadings_used.T
        
        # Rescale back to original values
        mean_spreads = spreads_df.mean()
        std_spreads = spreads_df.std()
        reconstructed_spreads = pd.DataFrame(
            reconstructed_scaled * std_spreads.values + mean_spreads.values,
            index=spreads_df.index,
            columns=spreads_df.columns
        )

        # 2. Reconstruct Outright Prices, Spreads, and Flies
        historical_outrights_df, historical_spreads_df, historical_butterflies_df = \
            reconstruct_prices_and_derivatives(generic_df, reconstructed_spreads, spreads_df, butterflies_df)


        # --- New Plotting Section ---
        st.header("4. Historical PCA Fair Curve Analysis")
        st.markdown(f"The plots below show the historical path of the original market data (Outrights, Spreads, and Butterflies) compared to the PCA-reconstructed 'Fair' paths using the selected **{pc_count}** Principal Components.")

        def plot_historical_comparison(df, title, y_label):
            fig, ax = plt.subplots(figsize=(15, 7))
            
            original_cols = [col for col in df.columns if '(Original)' in col]
            pca_cols = [col for col in df.columns if '(PCA)' in col]
            
            # Plot Original Data (thin, opaque)
            df[original_cols].plot(ax=ax, legend=False, linestyle='-', alpha=0.3, color='gray', linewidth=1.5)
            
            # Plot PCA Reconstructed Data (thicker, dashed)
            df[pca_cols].plot(ax=ax, legend=False, linestyle='--', linewidth=2.5)

            # Re-plot one original column with legend for clarity
            df[original_cols[0]].plot(ax=ax, linestyle='-', color='gray', label='Original Data (All lines)', alpha=0.7, linewidth=1.5)

            # Setup legend for PCA lines and one original line
            # The PCA lines are the last ones plotted
            pca_legend_lines = ax.lines[-len(pca_cols):]
            # The single original line is the one before the PCA group
            original_legend_line = ax.lines[-len(pca_cols) - 1]
            
            all_lines = [original_legend_line] + pca_legend_lines
            all_labels = ['Original Data'] + [col.replace(' (PCA)', '') for col in pca_cols]
            
            ax.legend(all_lines, all_labels, loc='best', ncol=3)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle=':', alpha=0.6)
            return fig

        # 4.1 Outrights Plot
        st.subheader("4.1 Historical Outright Prices")
        fig_outrights = plot_historical_comparison(
            historical_outrights_df, 
            'Historical Outright Prices: Original vs. PCA Fair Curve', 
            'Price'
        )
        st.pyplot(fig_outrights)

        # 4.2 Spreads Plot
        st.subheader("4.2 Historical Spreads")
        fig_spreads = plot_historical_comparison(
            historical_spreads_df, 
            'Historical Spreads: Original vs. PCA Reconstructed', 
            'Spread Value (Price Difference)'
        )
        st.pyplot(fig_spreads)

        # 4.3 Flies Plot
        if not historical_butterflies_df.empty:
            st.subheader("4.3 Historical Butterflies (Flies)")
            fig_flies = plot_historical_comparison(
                historical_butterflies_df, 
                'Historical Butterflies: Original vs. PCA Reconstructed', 
                'Fly Value'
            )
            st.pyplot(fig_flies)
        else:
            st.info("Not enough contracts (need 3 or more generic contracts, Q1, Q2, Q3) to calculate and plot butterflies.")
            
    else:
        st.error("PCA failed. Please check your data quantity and quality.")

# --- Mock Data Generation (For Initial Demonstration) ---
if price_file is None:
    st.subheader("Instructions and Example Data Format")
    st.markdown("To run the application, you need to provide two CSV files with the following formats. Example data is pre-populated for demonstration.")

    # Generate Mock Price Data
    dates = pd.to_datetime(pd.date_range(start=datetime(2024, 1, 1), end=datetime(2024, 6, 30), freq='D'))
    contracts = ['U4', 'Z4', 'H5', 'M5', 'U5', 'Z5', 'H6', 'M6']
    data = {}
    
    # Create realistic-looking data where level, slope, and curve factors are present
    rng = np.random.default_rng(42)
    level_factor = rng.normal(0, 0.1, size=len(dates)).cumsum()
    slope_factor = rng.normal(0, 0.05, size=len(dates)).cumsum()
    curve_factor = rng.normal(0, 0.01, size=len(dates)) * np.sin(np.linspace(0, 2*np.pi, len(dates)) * 5)
    
    base_levels = np.array([95.0, 94.8, 94.7, 94.6, 94.5, 94.4, 94.3, 94.2])
    
    for i, contract in enumerate(contracts):
        # Base Level + Level Factor + Slope Factor * i + Curve Factor * (i^2)
        prices = base_levels[i] + level_factor + slope_factor * (i * 0.5) + curve_factor * (i - 3)**2 * 0.05
        data[contract] = prices
        
    mock_price_df = pd.DataFrame(data, index=dates)
    mock_price_df.index.name = 'Date'
    
    # Generate Mock Expiry Data
    mock_expiry_data = {
        'Contract': ['U4', 'Z4', 'H5', 'M5', 'U5', 'Z5', 'H6', 'M6'],
        'ExpiryDate': [
            '2024-09-18', '2024-12-18', '2025-03-19', '2025-06-18', 
            '2025-09-17', '2025-12-17', '2026-03-18', '2026-06-17'
        ]
    }
    mock_expiry_df = pd.DataFrame(mock_expiry_data)

    st.markdown("### Example Price Data Format")
    st.code(mock_price_df.head().to_csv(sep='\t'), language='text')

    st.markdown("### Example Expiry Data Format")
    st.code(mock_expiry_df.head().to_csv(sep='\t', index=False), language='text')
    
    st.markdown("Please provide your actual data to proceed with the PCA analysis.")
