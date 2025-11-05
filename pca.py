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
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file content to inspect the header for format identification
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # --- Case 1: Expiry File (EXPIRY (2).csv format: MATURITY, DATE) ---
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            # Ensure Contract is the index and Date is datetime
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # --- Case 2: Price File (sofr rates.csv format: Date as index) ---
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True,
            sep=',', # Explicitly specify comma as separator
            header=0 # Ensure the first row is used as the header
        )
        
        df.index.name = 'Date'
        
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')
        
        # Convert all price columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Filter out any remaining rows where the index date is NaT or the row is entirely NaN
        df = df.dropna(how='all')
        df = df[df.index.notna()]

        if df.empty or df.shape[1] == 0:
             raise ValueError("DataFrame is empty after processing or has no data columns.")
             
        return df
        
    except Exception as e:
        st.error(f"Error loading and processing data from {uploaded_file.name}: {e}")
        return None


def get_analysis_contracts(expiry_df, analysis_date):
    """
    Filters contract codes that expire on or after the analysis date and returns
    them in chronological order with their expiry date. This list defines the 
    maturities used in the analysis curve (e.g., Z20, H21, M21...).
    """
    if expiry_df is None:
        return pd.DataFrame()
    
    # Filter contracts that expire on or after the analysis date
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    # Sort by expiry date
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    return future_expiries

def transform_to_analysis_curve(price_df, future_expiries_df):
    """
    Transforms the price DataFrame to only include relevant contracts in maturity order.
    The column names are kept as the original contract codes (e.g., Z20, H21).
    """
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []

    contract_order = future_expiries_df.index.tolist()
    
    # Filter price columns to only include those present in the contract order
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []

    # Filter the price data to only include valid, ordered contract columns
    analysis_curve_df = price_df[valid_contracts]
    
    return analysis_curve_df, valid_contracts

def calculate_outright_spreads(analysis_curve_df):
    """
    Calculates the first differences (spreads) on a CME basis: C1 - C2, C2 - C3, etc.
    The labels now use the contract codes (e.g., Z20-H21).
    """
    if analysis_curve_df.empty:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - 1):
        # CME Basis: Shorter maturity minus longer maturity
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+1]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+1]
        
    return pd.DataFrame(spreads_data)

def calculate_butterflies(analysis_curve_df):
    """
    Calculates butterflies (flies) on a CME basis: (C1 - C2) - (C2 - C3) = C1 - 2*C2 + C3, etc.
    The labels now use the contract codes (e.g., Z20-2xH21+M21).
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 3:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2):
        short_maturity = analysis_curve_df.columns[i]    # C1
        center_maturity = analysis_curve_df.columns[i+1] # C2
        long_maturity = analysis_curve_df.columns[i+2]   # C3

        # Fly = C1 - 2*C2 + C3
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = analysis_curve_df.iloc[:, i] - 2 * analysis_curve_df.iloc[:, i+1] + analysis_curve_df.iloc[:, i+2]

    return pd.DataFrame(flies_data)

def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads)."""
    # Drop rows with NaNs before standardization and PCA
    data_df_clean = data_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        st.error("Not enough complete data points (rows) to perform PCA on the spreads after dropping NaNs.")
        return None, None, None, None

    # Standardize the data (PCA is sensitive to scale)
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    data_scaled = (data_df_clean - data_mean) / data_std
    
    # Determine optimal number of components (min(n_samples, n_features))
    n_components = min(data_scaled.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Component Loadings (the eigenvectors * sqrt(eigenvalues))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    # Principal Component Scores (the transformed data)
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance, scores, data_df_clean

def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_df, spreads_df, butterflies_df):
    """
    Reconstructs Outright Prices and Derivatives historically using reconstructed spreads,
    anchored to the original nearest contract price path (Level factor).
    """
    # Filter the analysis_curve_df to match the index of the reconstructed spreads
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_df.index]
    
    # --- 1. Reconstruct Outright Prices ---
    
    # Anchor the entire curve reconstruction to the original historical nearest contract price path
    nearest_contract_original = analysis_curve_df_aligned.iloc[:, 0]
    nearest_contract_label = analysis_curve_df_aligned.columns[0]
    
    # Initialize the reconstructed prices DataFrame, starting with the original as the Level anchor
    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original
    
    # Iterate through all maturities starting from the second contract (index 1)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i-1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label = f"{prev_maturity}-{current_maturity}"
        
        # Calculate the reconstructed price P_i using P_i-1 (PCA) and S_i-1,i (PCA)
        # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
        reconstructed_prices_df[current_maturity + ' (PCA)'] = (
            reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_df[spread_label]
        )
        
    # Merge original prices for comparison
    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Prepare Spreads for comparison ---
    spreads_df_aligned = spreads_df.loc[reconstructed_spreads_df.index]
    original_spread_rename = {col: col + ' (Original)' for col in spreads_df_aligned.columns}
    pca_spread_rename = {col: col + ' (PCA)' for col in reconstructed_spreads_df.columns}

    original_spreads = spreads_df_aligned.rename(columns=original_spread_rename)
    pca_spreads = reconstructed_spreads_df.rename(columns=pca_spread_rename)
    
    historical_spreads = pd.merge(original_spreads, pca_spreads, left_index=True, right_index=True)
    
    
    # --- 3. Reconstruct Butterflies ---
    if butterflies_df.empty:
        return historical_outrights, historical_spreads, pd.DataFrame()
    
    butterflies_df_aligned = butterflies_df.loc[reconstructed_spreads_df.index]
        
    reconstructed_butterflies = {}
    for i in range(len(spreads_df.columns) - 1):
        spread1_label = spreads_df.columns[i]
        spread2_label = spreads_df.columns[i+1]
        original_fly_label = butterflies_df.columns[i]
        
        # Reconstruct fly: Fly = Spread1_PCA - Spread2_PCA
        reconstructed_butterflies[original_fly_label + ' (PCA)'] = (
            reconstructed_spreads_df[spread1_label] - reconstructed_spreads_df[spread2_label]
        )

    reconstructed_butterflies_df = pd.DataFrame(reconstructed_butterflies, index=reconstructed_spreads_df.index)
    
    # Merge original flies for comparison
    original_fly_rename = {col: col + ' (Original)' for col in butterflies_df_aligned.columns}
    original_butterflies_df = butterflies_df_aligned.rename(columns=original_fly_rename)
    
    historical_butterflies = pd.merge(original_butterflies_df, reconstructed_butterflies_df, left_index=True, right_index=True)
    
    return historical_outrights, historical_spreads, historical_butterflies


# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (e.g., 'sofr rates.csv')", 
    type=['csv'], 
    key='price_upload'
)
expiry_file = st.sidebar.file_uploader(
    "Upload Contract Expiry Dates (e.g., 'EXPIRY (2).csv')", 
    type=['csv'], 
    key='expiry_upload'
)

# Initialize dataframes
price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

if price_df is not None and expiry_df is not None:
    # --- Date Range Filter ---
    st.sidebar.header("2. Historical Date Range")
    min_date = price_df.index.min().date()
    max_date = price_df.index.max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Historical Data Range for PCA Calibration", 
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    price_df_filtered = price_df[(price_df.index.date >= start_date) & (price_df.index.date <= end_date)]
    
    # --- Analysis Date Selector (Maturity Roll) ---
    st.sidebar.header("3. Curve Analysis Date")
    
    # Analysis date should be within the historical range for stability
    default_analysis_date = end_date
    if default_analysis_date < min_date:
        default_analysis_date = min_date
        
    analysis_date = st.sidebar.date_input(
        "Select **Single Date** for Curve Snapshot", 
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
if not price_df_filtered.empty:
    
    # 1. Get the list of relevant contracts based on the analysis date
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
    
    if future_expiries_df.empty:
        st.warning("Could not establish a relevant contract curve. Please check your date filters.")
        st.stop()
        
    # 2. Transform historical prices to the required maturity curve
    analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)

    if analysis_curve_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Spreads and Butterflies (Inputs for PCA and comparison)
    st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    
    # Calculate Spreads
    spreads_df = calculate_outright_spreads(analysis_curve_df)
    st.markdown("##### Outright Spreads (e.g., Z20-H21, H21-M21, etc.)")
    st.dataframe(spreads_df.head(5))
    
    if spreads_df.empty:
        st.warning("Spreads could not be calculated. Need at least two contracts in the analysis curve.")
        st.stop()
        
    # Calculate Butterflies
    butterflies_df = calculate_butterflies(analysis_curve_df)
    st.markdown("##### Butterflies (e.g., Z20-2xH21+M21, etc.)")
    st.dataframe(butterflies_df.head(5))

    # 4. Perform PCA on Spreads
    # PCA is performed only on spreads, as they are the most stationary features
    loadings, explained_variance, scores, spreads_df_clean = perform_pca(spreads_df)

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
                help="Typically, the first 3 components (Level, Slope, Curve) explain over 95% of variance in spread changes."
            )
            
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads.")
        
        
        # --- Component Loadings Heatmap ---
        st.header("3. PC Loadings Heatmap (PC vs. Spreads)")
        st.markdown("""
            This heatmap shows the weights of the first few PCs on each **Spread**. These weights define the fundamental Level, Slope, and Curvature factors.
        """)
        
        plt.style.use('default') 
        fig, ax = plt.subplots(figsize=(12, 6))
        
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
        ax.set_ylabel('Spread Contract')
        st.pyplot(fig)
        
        
        # --- PC Scores Time Series Plot ---
        def plot_pc_scores(scores_df, explained_variance):
            """Plots the time series of the first 3 PC scores."""
            
            pc_labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
            num_pcs = min(3, scores_df.shape[1])
            if num_pcs == 0: return None

            fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
            if num_pcs == 1: axes = [axes] 

            plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)

            for i in range(num_pcs):
                ax = axes[i]
                pc_label = pc_labels[i]
                variance_pct = explained_variance[i] * 100
                
                ax.plot(scores_df.index, scores_df.iloc[:, i], label=f'{pc_label} ({variance_pct:.2f}% Var.)', linewidth=1.5, color=plt.cm.tab10(i))
                
                ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
                
                ax.set_title(f'{pc_label} Factor Score (Explaining {variance_pct:.2f}% of Spread Variance)', fontsize=14)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_ylabel('Score Value')
                ax.legend(loc='upper left')
                
            plt.xlabel('Date')
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            return fig

        st.header("4. PC Factor Scores Time Series")
        st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range.")
        fig_scores = plot_pc_scores(scores, explained_variance)
        if fig_scores:
            st.pyplot(fig_scores)
            
        
        # --- Historical Reconstruction ---
        
        # 1. Reconstruct Spreads using only selected PCs
        data_mean = spreads_df_clean.mean()
        data_std = spreads_df_clean.std()
        scores_used = scores.values[:, :pc_count]
        loadings_used = loadings.values[:, :pc_count]
        
        reconstructed_scaled = scores_used @ loadings_used.T
        
        reconstructed_spreads = pd.DataFrame(
            reconstructed_scaled * data_std.values + data_mean.values,
            index=spreads_df_clean.index, 
            columns=spreads_df_clean.columns
        )

        # 2. Reconstruct Outright Prices, Spreads, and Flies
        historical_outrights_df, historical_spreads_df, historical_butterflies_df = \
            reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads, spreads_df, butterflies_df)

        
        
        # --- NEW: Cross-Sectional Curve Plot for Single Date ---
        st.header("5. Curve Snapshot Analysis: " + analysis_date.strftime('%Y-%m-%d'))
        st.markdown("This chart plots the **current market curve** (Original) against the **PCA Fair curve** for the selected date. The vertical difference is the theoretical mispricing.")

        # Get the single-day snapshot
        try:
            curve_snapshot_original = historical_outrights_df.filter(regex='\(Original\)$').loc[analysis_dt].T.rename(columns={analysis_dt: 'Original'}).droplevel(1)
            curve_snapshot_pca = historical_outrights_df.filter(regex='\(PCA\)$').loc[analysis_dt].T.rename(columns={analysis_dt: 'PCA Fair'}).droplevel(1)
            
            curve_comparison = pd.concat([curve_snapshot_original, curve_snapshot_pca], axis=1).dropna()
            
            if curve_comparison.empty:
                st.warning(f"No complete data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')}.")
            else:
                # --- Plot the Curve ---
                fig_curve, ax_curve = plt.subplots(figsize=(15, 7))
                
                # Plot Original Curve
                ax_curve.plot(curve_comparison.index, curve_comparison['Original'], 
                              label='Original Market Curve', marker='o', linestyle='-', linewidth=2.5, color='blue')
                
                # Plot PCA Fair Curve
                ax_curve.plot(curve_comparison.index, curve_comparison['PCA Fair'], 
                              label=f'PCA Fair Curve ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                # Plot Mispricing (Original - PCA Fair)
                mispricing = curve_comparison['Original'] - curve_comparison['PCA Fair']
                
                # Annotate the contracts with the largest absolute mispricing
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 10000 # Convert to BPS
                    
                    ax_curve.annotate(
                        f"Mispricing: {mispricing_value:.2f} BPS",
                        (mispricing_contract, curve_comparison.loc[mispricing_contract]['Original']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )
                
                ax_curve.set_title(f'Market Price Curve vs. PCA Fair Price Curve on {analysis_date.strftime("%Y-%m-%d")}', fontsize=16)
                ax_curve.set_xlabel('Contract Maturity')
                ax_curve.set_ylabel('Price (100 - Rate)')
                ax_curve.legend(loc='upper right')
                ax_curve.grid(True, linestyle=':', alpha=0.6)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_curve)
                
                # --- Detailed Contract Price/Rate Table (New Feature) ---
                st.subheader("5.1 Detailed Contract Prices and Mispricing")
                
                # Create the detailed table
                detailed_comparison = curve_comparison.copy()
                detailed_comparison.index.name = 'Contract'
                
                # Calculate Rates (Yields) and Mispricing in BPS
                detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
                detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
                # Mispricing in Price terms (Original Price - PCA Fair Price) * 10,000 to get BPS
                detailed_comparison['Mispricing (BPS)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 10000

                # Reorder columns and rename Price columns
                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'}
                )
                
                detailed_comparison = detailed_comparison[[
                    'Original Price', 
                    'Original Rate (%)', 
                    'PCA Fair Price', 
                    'PCA Fair Rate (%)', 
                    'Mispricing (BPS)'
                ]]
                
                # Display the table, formatted for financial data
                st.dataframe(
                    detailed_comparison.style.format({
                        'Original Price': "{:.4f}",
                        'PCA Fair Price': "{:.4f}",
                        'Original Rate (%)': "{:.4f}",
                        'PCA Fair Rate (%)': "{:.4f}",
                        'Mispricing (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
        except KeyError:
            st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data. Please choose a different date within the historical range.")
        
        # --- Historical Time Series Comparison ---
        st.header("6. Historical Time Series Comparison")
        st.markdown(f"These plots compare the historical path of the market data against the PCA-reconstructed 'Fair' paths using the selected **{pc_count}** Principal Components.")


        def plot_historical_comparison(df, title, y_label):
            df = df.dropna() 
            if df.empty:
                st.warning(f"No complete data points available to plot for {title}.")
                return None
                
            fig, ax = plt.subplots(figsize=(15, 7))
            
            original_cols = [col for col in df.columns if '(Original)' in col]
            pca_cols = [col for col in df.columns if '(PCA)' in col]
            
            # Plot Original Data (thin, dashed, gray/darker)
            df[original_cols].plot(ax=ax, legend=False, linestyle='-', alpha=0.4, color='darkgray', linewidth=1.5)
            
            # Plot PCA Reconstructed Data (thicker, solid, colored)
            df[pca_cols].plot(ax=ax, legend=False, linestyle='-', linewidth=2.5)

            # Re-plot one Original column for a clear legend entry
            if original_cols:
                original_legend_label = 'Original Market Data'
                df[original_cols[0]].plot(ax=ax, linestyle='--', color='black', alpha=0.6, label=original_legend_label, linewidth=1.5)

            # Setup legend for PCA lines and one original line
            if pca_cols:
                pca_legend_lines = ax.lines[-len(pca_cols):]
                original_line_for_legend = ax.lines[-len(pca_cols) - 1]
                
                all_lines = [original_line_for_legend] + pca_legend_lines
                all_labels = [original_legend_label] + [col.replace(' (PCA)', ' (PCA Fair)') for col in pca_cols]
                
                ax.legend(all_lines, all_labels, loc='best', ncol=4, fontsize='small')
                    
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle=':', alpha=0.6)
            return fig

        # 6.1 Outrights Plot
        st.subheader("6.1 Historical Outright Prices")
        fig_outrights = plot_historical_comparison(
            historical_outrights_df, 
            'Historical Outright Prices Time Series', 
            'Price'
        )
        if fig_outrights:
            st.pyplot(fig_outrights)

        # 6.2 Spreads Plot
        st.subheader("6.2 Historical Spreads")
        fig_spreads = plot_historical_comparison(
            historical_spreads_df, 
            'Historical Spreads Time Series', 
            'Spread Value (Price Difference)'
        )
        if fig_spreads:
            st.pyplot(fig_spreads)

        # 6.3 Flies Plot
        if not historical_butterflies_df.empty:
            st.subheader("6.3 Historical Butterflies (Flies)")
            fig_flies = plot_historical_comparison(
                historical_butterflies_df, 
                'Historical Butterflies Time Series', 
                'Fly Value'
            )
            if fig_flies:
                st.pyplot(fig_flies)
        else:
            st.info("Not enough contracts (need 3 or more) to calculate and plot butterflies.")
            
    else:
        st.error("PCA failed. Please check your data quantity and quality.")
