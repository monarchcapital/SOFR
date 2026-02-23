import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# --- PDF figure collections ---
SECTION5_FIGURES = []
SECTION9_FIGURES = []


# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer")

# --- Helper Functions for Data Processing ---

# Use st.cache_data for performance as file loading is idempotent
@st.cache_data
def load_data(uploaded_file):
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file content to inspect the header for format identification
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # --- Case 1: Expiry File (MATURITY, DATE) ---
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # --- Case 2: Price File (Date as index) ---
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True,
            sep=',', 
            header=0 
        )
        
        df.index.name = 'Date'
        df = df.dropna(axis=1, how='all')
        
        for col in df.columns:
            # Ensure price columns are numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(how='all')
        df = df[df.index.notna()]

        if df.empty or df.shape[1] == 0:
             raise ValueError("DataFrame is empty after processing or has no data columns.")
             
        return df
        
    except Exception as e:
        st.error(f"Error loading and processing data from {uploaded_file.name}: {e}")
        return None


@st.cache_data
def get_analysis_contracts(expiry_df, analysis_date):
    """Filters contract codes that expire on or after the analysis date."""
    if expiry_df is None:
        return pd.DataFrame()
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
    
    return future_expiries

@st.cache_data
def transform_to_analysis_curve(price_df, future_expiries_df):
    """Selects and orders historical prices for relevant contracts."""
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []
    contract_order = future_expiries_df.index.tolist()
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []
    analysis_curve_df = price_df[valid_contracts]
    return analysis_curve_df, valid_contracts


# --- GENERALIZED DERIVATIVE CALCULATION FUNCTIONS (k-step) ---

@st.cache_data
def calculate_k_step_spreads(analysis_curve_df, k):
    """
    Calculates spreads between contracts separated by 'k' steps (e.g., k=1 for 3M, k=2 for 6M, k=4 for 12M).
    CME Basis: C_i - C_{i+k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - k):
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+k]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        # Spread = C_i - C_{i+k}
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+k]
        
    return pd.DataFrame(spreads_data)

@st.cache_data
def calculate_k_step_butterflies(analysis_curve_df, k):
    """
    Calculates butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M fly, k=2 for 6M fly, k=4 for 12M fly).
    Formula: C_i - 2 * C_{i+k} + C_{i+2k}
    Label Format: C_i-2xC_{i+k}+C_{i+2k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 2 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2 * k):
        short_maturity = analysis_curve_df.columns[i]      # C_i
        center_maturity = analysis_curve_df.columns[i+k]   # C_{i+k}
        long_maturity = analysis_curve_df.columns[i+2*k]   # C_{i+2k}

        # Fly = C_i - 2*C_{i+k} + C_{i+2k}
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = analysis_curve_df.iloc[:, i] - 2 * analysis_curve_df.iloc[:, i+k] + analysis_curve_df.iloc[:, i+2*k]

    return pd.DataFrame(flies_data)

# --- Double Butterfly Calculation Function ---
@st.cache_data
def calculate_k_step_double_butterflies(analysis_curve_df, k):
    """
    Calculates double butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M DBF).
    Formula: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}
    Label Format: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
    """
    # Need 4 contracts: C_i, C_{i+k}, C_{i+2k}, C_{i+3k}
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 3 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    dbflies_data = {}

    for i in range(num_contracts - 3 * k):
        c1_maturity = analysis_curve_df.columns[i]          # C_i
        c2_maturity = analysis_curve_df.columns[i+k]        # C_{i+k}
        c3_maturity = analysis_curve_df.columns[i+2*k]      # C_{i+2k}
        c4_maturity = analysis_curve_df.columns[i+3*k]      # C_{i+3k}

        # DBF = C_i - 3*C_{i+k} + 3*C_{i+2k} - C_{i+3k}
        dbfly_label = f"{c1_maturity}-3x{c2_maturity}+3x{c3_maturity}-{c4_maturity}"

        dbflies_data[dbfly_label] = (
            analysis_curve_df.iloc[:, i] 
            - 3 * analysis_curve_df.iloc[:, i+k] 
            + 3 * analysis_curve_df.iloc[:, i+2*k] 
            - analysis_curve_df.iloc[:, i+3*k]
        )

    return pd.DataFrame(dbflies_data)


def compute_all_derivatives_from_outrights_row(contract_labels, outrights_row):
    """Given a single outright curve (Series indexed by contract labels),
    compute all 3M/6M/12M spreads, flies, and double flies for that snapshot.

    This is used in the PCA shock engine to rebuild **all** curve derivatives
    from a shocked outright curve in a consistent way.
    """
    contracts = list(contract_labels)
    n = len(contracts)

    def _compute_for_k(k):
        spreads = {}
        flies = {}
        dbflies = {}

        # Spreads: C_i - C_{i+k}
        for i in range(n - k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            spreads[f"{c1}-{c2}"] = outrights_row[c1] - outrights_row[c2]

        # Flies: C_i - 2*C_{i+k} + C_{i+2k}
        for i in range(n - 2 * k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            c3 = contracts[i + 2 * k]
            flies[f"{c1}-2x{c2}+{c3}"] = (
                outrights_row[c1]
                - 2 * outrights_row[c2]
                + outrights_row[c3]
            )

        # Double Flies: C_i - 3*C_{i+k} + 3*C_{i+2k} - C_{i+3k}
        for i in range(n - 3 * k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            c3 = contracts[i + 2 * k]
            c4 = contracts[i + 3 * k]
            dbflies[f"{c1}-3x{c2}+3x{c3}-{c4}"] = (
                outrights_row[c1]
                - 3 * outrights_row[c2]
                + 3 * outrights_row[c3]
                - outrights_row[c4]
            )

        return (
            pd.Series(spreads) if spreads else pd.Series(dtype=float),
            pd.Series(flies) if flies else pd.Series(dtype=float),
            pd.Series(dbflies) if dbflies else pd.Series(dtype=float),
        )

    # 3M (k=1), 6M (k=2), 12M (k=4)
    spreads_3M, flies_3M, dbf_3M = _compute_for_k(1)
    spreads_6M, flies_6M, dbf_6M = _compute_for_k(2)
    spreads_12M, flies_12M, dbf_12M = _compute_for_k(4)

    return {
        "3M_spreads": spreads_3M,
        "3M_flies": flies_3M,
        "3M_dbf": dbf_3M,
        "6M_spreads": spreads_6M,
        "6M_flies": flies_6M,
        "6M_dbf": dbf_6M,
        "12M_spreads": spreads_12M,
        "12M_flies": flies_12M,
        "12M_dbf": dbf_12M,
    }


# --- END GENERALIZED DERIVATIVE CALCULATION FUNCTIONS ---


def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads for Fair Curve)."""
    data_df_clean = data_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None, None, None, None

    # Standardize the data (PCA on Correlation Matrix - preferred for spread PCA)
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    data_scaled = (data_df_clean - data_mean) / data_std
    
    n_components = min(data_scaled.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Loadings (Eigenvectors on Correlation Matrix)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    # Get Eigenvalues (Variance of the principal components)
    eigenvalues = pca.explained_variance_
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance_ratio, eigenvalues, scores, data_df_clean

# --- PCA ON PRICES (FOR NON-UNIFORM PC1 VISUALIZATION) ---
def perform_pca_on_prices(price_df):
    """
    Performs PCA directly on Outright Price Levels using the COVARIANCE MATRIX 
    (unstandardized data), which results in a NON-UNIFORM PC1.
    """
    data_df_clean = price_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None
        
    # Center the data, but DO NOT scale/standardize it (PCA on Covariance Matrix)
    data_centered = data_df_clean - data_df_clean.mean() 
    
    n_components = min(data_centered.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    
    # Loadings (Eigenvectors - the raw sensitivities)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    return loadings, explained_variance

# --- RECONSTRUCTION LOGIC ---

def _reconstruct_derivative(original_df, reconstructed_prices, derivative_type='spread'):
    """
    Helper to reconstruct a derivative from the reconstructed price curve.
    """
    if original_df.empty:
        return pd.DataFrame()

    # Align the original data index with the reconstructed prices index
    valid_indices = reconstructed_prices.index.intersection(original_df.index)
    original_df_aligned = original_df.loc[valid_indices]
    reconstructed_prices_aligned = reconstructed_prices.loc[valid_indices]
    
    reconstructed_data = {}
    
    for label in original_df_aligned.columns:
        
        try:
            if derivative_type == 'spread':
                # Spread: C_i - C_{i+k}. Label is X Spread: C_i-C_{i+k} (e.g., 3M Spread: Z25-M26)
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                c1, c_long = core_label.split('-')
                
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - reconstructed_prices_aligned[c_long + ' (PCA)']
                )
            
            elif derivative_type == 'fly':
                # Fly: C_i - 2 * C_{i+k} + C_{i+2k}. Label format: X Fly: C_i-2xC_{i+k}+C_{i+2k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] 
                sub_parts = parts[1].split('+')
                c2_label = sub_parts[0].split('x')[1] 
                c3_label = sub_parts[1] 
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    2 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    reconstructed_prices_aligned[c3_label + ' (PCA)']
                )
            
            elif derivative_type == 'dbfly':
                # Double Fly: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}. Label format: X Double Fly: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] # C_i
                
                sub_parts_1 = parts[1].split('+')
                
                c2_label = sub_parts_1[0].split('x')[1] # C_{i+k} from '3xC_{i+k}'
                
                sub_parts_2 = sub_parts_1[1].split('-')
                
                c3_label = sub_parts_2[0].split('x')[1] # C_{i+2k} from '3xC_{i+2k}'
                c4_label = sub_parts_2[1] # C_{i+3k}
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    3 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    3 * reconstructed_prices_aligned[c3_label + ' (PCA)'] -
                    reconstructed_prices_aligned[c4_label + ' (PCA)']
                )
            
        except Exception as e:
             # Skip if reconstruction fails due to malformed label or missing price
             continue 
    
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices_aligned.index)
    
    original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
    original_df_renamed = original_df_aligned.rename(columns=original_rename)
    
    return pd.merge(original_df_renamed, reconstructed_df, left_index=True, right_index=True)


def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df):
    """
    Reconstructs Outright Prices and all derivative types based on the 
    reconstructed 3M spreads (PCA result) and the original nearest contract price anchor.
    """
    # Filter the analysis_curve_df to match the index of the reconstructed 3M spreads
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_3M_df.index]
    
    # --- 1. Reconstruct Outright Prices (Anchor) ---
    nearest_contract_original = analysis_curve_df_aligned.iloc[:, 0]
    nearest_contract_label = analysis_curve_df_aligned.columns[0]
    
    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original # Anchor
    
    spreads_3M_df_no_prefix = spreads_3M_df.copy()

    # Reconstruct all subsequent contracts using the reconstructed 3M spreads (k=1)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i-1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label_no_prefix = f"{prev_maturity}-{current_maturity}" # This is always the 3M spread label

        spread_label_reconstructed = f"3M Spread: {spread_label_no_prefix}"
        
        if spread_label_no_prefix in reconstructed_spreads_3M_df.columns:
            # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_3M_df[spread_label_no_prefix]
            )
        else:
            # Fallback if the 3M spread is missing for that contract roll
             reconstructed_prices_df[current_maturity + ' (PCA)'] = reconstructed_prices_df[prev_maturity + ' (PCA)']

    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Reconstruct Derivatives from Reconstructed Prices ---
    
    # Prepare derivative DFs with prefixes for _reconstruct_derivative to correctly rename columns
    spreads_3M_df_prefixed = spreads_3M_df_no_prefix.rename(columns=lambda x: f"3M Spread: {x}")
    butterflies_3M_df_prefixed = butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}")
    spreads_6M_df_prefixed = spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}")
    butterflies_6M_df_prefixed = butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}")
    spreads_12M_df_prefixed = spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}")
    butterflies_12M_df_prefixed = butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}")
    
    # New Double Butterfly DFs
    double_butterflies_3M_df_prefixed = double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}")
    double_butterflies_6M_df_prefixed = double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}")
    double_butterflies_12M_df_prefixed = double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}")

    historical_spreads_3M = _reconstruct_derivative(spreads_3M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_6M = _reconstruct_derivative(spreads_6M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_12M = _reconstruct_derivative(spreads_12M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    # New Double Butterfly reconstructions
    historical_double_butterflies_3M = _reconstruct_derivative(double_butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_6M = _reconstruct_derivative(double_butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_12M = _reconstruct_derivative(double_butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')

    # MODIFIED: Return the new historical double butterfly DFs
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M, historical_double_butterflies_3M, historical_double_butterflies_6M, historical_double_butterflies_12M, spreads_3M_df_no_prefix


# --- ORIGINAL HEDGING LOGIC (Section 6) ---

def calculate_reconstructed_covariance(loadings_df, eigenvalues, spread_std_dev, pc_count):
    """
    Calculates the covariance matrix of the STANDARDIZED spreads 
    reconstructed using the first 'pc_count' PCs: Sigma_scaled = L_p Lambda_p L_p^T
    Then scales back to original spread space: Sigma = (diag(sigma)) * Sigma_scaled * (diag(sigma))
    """
    # 1. Select the loadings and eigenvalues for the used PCs
    L_p = loadings_df.iloc[:, :pc_count].values # Loadings (Eigenvectors on Correlation Matrix)
    lambda_p = eigenvalues[:pc_count]           # Eigenvalues (Variance of standardized scores)
    
    # 2. Reconstruct the Covariance Matrix of the Standardized Data
    # Sigma_scaled = L_p * Lambda_p * L_p^T
    Sigma_scaled = L_p @ np.diag(lambda_p) @ L_p.T
    
    # 3. Scale back to the original spread data covariance matrix
    # Cov(X) = diag(sigma) * Cov(Z) * diag(sigma)
    Sigma = Sigma_scaled * np.outer(spread_std_dev.values, spread_std_dev.values)
    
    Sigma_df = pd.DataFrame(Sigma, index=loadings_df.index, columns=loadings_df.index)
    
    return Sigma_df

def calculate_best_and_worst_hedge_3M(trade_label, loadings_df, eigenvalues, pc_count, spreads_3M_df_clean):
    """
    Calculates the best (min residual risk) and worst (max residual risk) 
    hedge for a given 3M spread trade using the reconstructed covariance matrix, 
    and returns the full results DataFrame as well. (Section 6 - 3M Spreads only)
    """
    if trade_label not in loadings_df.index:
        return None, None, None
        
    spread_std_dev = spreads_3M_df_clean.std()
    
    # Reconstruct covariance matrix using selected PCs
    Sigma_reconstructed = calculate_reconstructed_covariance(
        loadings_df, eigenvalues, spread_std_dev, pc_count
    )
    
    trade_spread = trade_label
    
    results = []
    
    # Iterate through all other 3M spreads as potential hedges
    potential_hedges = [col for col in Sigma_reconstructed.columns if col != trade_spread]
    
    for hedge_spread in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_reconstructed.loc[trade_spread, trade_spread] # Var(T)
        Var_Hedge = Sigma_reconstructed.loc[hedge_spread, hedge_spread] # Var(H)
        Cov_TH = Sigma_reconstructed.loc[trade_spread, hedge_spread]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge == 0:
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance of the hedged portfolio (Var(T - k*H) = Var(T) - k*Cov(T,H))
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility (Score) in Rate % (was BPS, now divided by 100)
        # 1 point = 100 BPS = 1% Rate
        Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100 # MODIFIED: * 10000 -> * 100
        
        results.append({
            'Hedge Spread': hedge_spread,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct # MODIFIED: Column name update
        })

    if not results:
        return None, None, None
        
    results_df = pd.DataFrame(results)
    
    # Best hedge minimizes Residual Volatility
    best_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True).iloc[0]
    
    # Worst hedge maximizes Residual Volatility
    worst_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=False).iloc[0]
    
    # Return the individual best/worst series AND the full DataFrame
    return best_hedge, worst_hedge, results_df


# --- GENERALIZED HEDGING LOGIC (Section 7) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix for ALL derivatives (Spreads, Flies, Double Flies) 
    by projecting their standardized time series onto the standardized 3M Spread PC scores.
    Returns the Raw Covariance Matrix, the aligned derivatives data, and the standardized loadings (L_D).
    """
    # 1. Align and clean data - ensure all derivatives are aligned with the PC scores index
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    
    if derivatives_aligned.empty:
        # Return empty dataframes, but return all three expected variables
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
        
    # 2. Standardize all derivatives
    derivatives_mean = derivatives_aligned.mean()
    derivatives_std = derivatives_aligned.std()
    # Subtract mean is important for proper regression/loadings calculation
    derivatives_scaled = (derivatives_aligned - derivatives_mean) / derivatives_std
    
    # 3. Calculate Loadings (Beta) of each standardized derivative on the standardized PCs
    
    loadings_data = {}
    X = scores_aligned.iloc[:, :pc_count].values # Standardized PC scores
    
    # Use Linear Regression to find the standardized loading (beta) for each derivative
    for col in derivatives_scaled.columns:
        y = derivatives_scaled[col].values
        # Using intercept=False as both X (scores) and y (scaled derivative) are mean-zero
        reg = LinearRegression(fit_intercept=False) 
        reg.fit(X, y)
        loadings_data[col] = reg.coef_

    # L_D: Loadings of the full derivatives set D onto the PC space
    loadings_df = pd.DataFrame(
        loadings_data, 
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T
    
    # 4. Reconstruct the Covariance Matrix in Standardized Space
    # Sigma_Std = L_D * Lambda_p * L_D^T
    L_D = loadings_df.values
    lambda_p = eigenvalues[:pc_count]
    Sigma_Std = L_D @ np.diag(lambda_p) @ L_D.T
    
    # 5. Scale back to the original derivative data covariance matrix (Raw Space)
    # Cov_Raw = diag(sigma) * Sigma_Std * diag(sigma)
    Sigma_Raw = Sigma_Std * np.outer(derivatives_std.values, derivatives_std.values)
    
    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=derivatives_aligned.columns, columns=derivatives_aligned.columns)
    
    return Sigma_Raw_df, derivatives_aligned, loadings_df

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    """
    Calculates the best/worst hedge using the generalized Raw Covariance Matrix (Sigma_Raw_df).
    (Section 7 - All Derivatives)
    """
    
    if trade_label not in Sigma_Raw_df.index:
        return None, None, None
        
    results = []
    
    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]
    
    for hedge_instrument in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)
        Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
        Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge <= 1e-9: # Check for near-zero variance
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance of the hedged portfolio (Var(T - k*H) = Var(T) - k*Cov(T,H))
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility (Score) in Rate % (was BPS, now divided by 100)
        # 1 point = 100 BPS = 1% Rate
        Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100 # MODIFIED: * 10000 -> * 100
        
        results.append({
            'Hedge Instrument': hedge_instrument,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct # MODIFIED: Column name update
        })

    if not results:
        return None, None, None
        
    results_df = pd.DataFrame(results)
    
    # Best hedge minimizes Residual Volatility
    best_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True).iloc[0]
    
    # Worst hedge maximizes Residual Volatility
    worst_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=False).iloc[0]
    
    # Return the individual best/worst series AND the full DataFrame
    return best_hedge, worst_hedge, results_df

# --- FACTOR-BASED HEDGING LOGIC (Section 8) ---

def calculate_factor_sensitivities(loadings_df_gen, pc_count):
    """
    Calculates the Standardized Sensitivity (Beta) of every derivative to the first three 
    principal components (Level, Slope, Curvature).
    """
    if loadings_df_gen.empty:
        return pd.DataFrame()

    # Define the factor mapping based on the first 3 PCs
    pc_map = {
        'PC1': 'Level (Whole Curve Shift)', 
        'PC2': 'Slope (Steepening/Flattening)', 
        'PC3': 'Curvature (Fly Risk)'
    }
    
    # Only use up to the number of available PCs, or 3, whichever is smaller
    available_pcs = loadings_df_gen.columns.intersection(list(pc_map.keys()))
    
    # Filter the generalized loadings L_D for the relevant PCs
    factor_sensitivities = loadings_df_gen.filter(items=available_pcs.tolist(), axis=1).copy()
    
    # Rename columns for clarity in the output
    factor_sensitivities.columns = [pc_map[col] for col in available_pcs]
    
    return factor_sensitivities

# --- NEW FUNCTION FOR TRIPLE FACTOR NEUTRALIZATION CHECK ---
def find_perfect_factor_hedge(trade_label, factor_sensitivities_df, mispricing_series, pc_count, tolerance=1e-4):
    """
    Identifies a single hedge instrument that can simultaneously neutralize the first
    three principal components (Level, Slope, and Curvature) for a given trade.
    
    Returns a dictionary of results or None if no perfect hedge is found.
    """
    if trade_label not in factor_sensitivities_df.index:
        return {'error': f"Trade instrument '{trade_label}' not found in sensitivities.", 'result': None}
    
    available_factors = factor_sensitivities_df.columns.intersection(['Level (Whole Curve Shift)', 'Slope (Steepening/Flattening)', 'Curvature (Fly Risk)'])
    
    # Need at least 3 PCs for the triple neutralization check
    if len(available_factors) < 3:
        return {'error': f"Need at least 3 PCs (Level, Slope, Curvature) for triple neutralization check. Only {len(available_factors)} available.", 'result': None}
    
    # Filter trade sensitivities to the first three factors
    T_sens = factor_sensitivities_df.loc[trade_label, available_factors]
    
    if T_sens.abs().sum() < 1e-9:
        return {'error': "The trade itself has near-zero sensitivity to the first three factors, thus no hedging is needed for these factors.", 'result': None}

    potential_hedges = [col for col in factor_sensitivities_df.index if col != trade_label]
    
    # Placeholder for the best hedge found based on tolerance
    best_match_result = None
    
    for hedge_instrument in potential_hedges:
        H_sens = factor_sensitivities_df.loc[hedge_instrument, available_factors]
        
        # Check if any factor sensitivity for the hedge is near zero
        # If the hedge has near-zero exposure, the ratio blows up, so skip
        if (H_sens.abs() < 1e-9).any():
            continue
            
        # Calculate the three required hedge ratios: k = E(T) / E(H)
        k_ratios = T_sens / H_sens
        
        k1, k2, k3 = k_ratios.values # k_PC1, k_PC2, k_PC3

        # Check for near-equality of the ratios
        diff1 = abs(k1 - k2)
        diff2 = abs(k1 - k3)
        diff3 = abs(k2 - k3)
        
        max_k_diff = max(diff1, diff2, diff3)

        # Check if the largest difference is within tolerance
        if max_k_diff < tolerance:
            # Found a perfect hedge
            avg_k = k_ratios.mean()
            
            # Determine hedge action
            hedge_action = 'Short' if avg_k > 0 else 'Long'
            
            # Fetch mispricing
            hedge_mispricing = mispricing_series.get(hedge_instrument, np.nan) 
            
            result = {
                'Hedge Instrument': hedge_instrument,
                'Trade PC1 Sensitivity': T_sens.iloc[0],
                'Trade PC2 Sensitivity': T_sens.iloc[1],
                'Trade PC3 Sensitivity': T_sens.iloc[2],
                'Hedge PC1 Sensitivity': H_sens.iloc[0],
                'Hedge PC2 Sensitivity': H_sens.iloc[1],
                'Hedge PC3 Sensitivity': H_sens.iloc[2],
                'Hedge Ratio (|k|)': abs(avg_k),
                'Hedge Action': hedge_action,
                'Hedge Mispricing (Rate %)': hedge_mispricing,
                'Max K Difference': max_k_diff
            }
            # Since we found one, we store it and can break or continue to find the one with the smallest Max K Difference
            # For simplicity, we just return the first one found that meets the tolerance, but for robustness,
            # we should find the one with the minimum max_k_diff.
            if best_match_result is None or max_k_diff < best_match_result.get('Max K Difference', tolerance):
                best_match_result = result

    if best_match_result:
        return {'error': None, 'result': best_match_result}
    else:
        return {'error': f"No single hedge instrument was found to neutralize Level, Slope, and Curvature simultaneously within the tolerance of {tolerance:.0e}.", 'result': None}

# --- END NEW FUNCTION ---


def calculate_all_factor_hedges(trade_label, factor_name, factor_sensitivities_df, Sigma_Raw_df):
    """
    Calculates the Factor Hedge Ratio and the resulting Residual Volatility for all potential 
    hedge instruments, for a specified factor.
    """
    if trade_label not in factor_sensitivities_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in sensitivities."
    if factor_name not in factor_sensitivities_df.columns:
        return pd.DataFrame(), f"Factor '{factor_name}' not found."
    if trade_label not in Sigma_Raw_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in covariance matrix."

    results = []
    
    Trade_Exposure = factor_sensitivities_df.loc[trade_label, factor_name]
    Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)
    
    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]

    for hedge_instrument in potential_hedges:
        try:
            Hedge_Exposure = factor_sensitivities_df.loc[hedge_instrument, factor_name]
            Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
            Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]        # Cov(T, H)

            # 1. Calculate Factor Hedge Ratio (k_factor)
            if abs(Hedge_Exposure) < 1e-9:
                k_factor = 0.0
                Residual_Volatility_Rate_Pct = np.nan # Cannot neutralize factor with zero-exposure hedge
            else:
                # k_factor is the ratio of sensitivities: k = Beta_T / Beta_H
                k_factor = Trade_Exposure / Hedge_Exposure
                
                # 2. Calculate Residual Variance of the hedged portfolio (Var(T - k*H))
                # Var(P) = Var(T) + k^2 Var(H) - 2k Cov(T, H)
                Residual_Variance = Var_Trade + (k_factor**2 * Var_Hedge) - (2 * k_factor * Cov_TH)
                Residual_Variance = max(0, Residual_Variance) 
                
                # 3. Residual Volatility (Score) in Rate % (was BPS, now divided by 100)
                # 1 point = 100 BPS = 1% Rate
                Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100 # MODIFIED: * 10000 -> * 100
                
            results.append({
                'Hedge Instrument': hedge_instrument,
                'Trade Sensitivity': Trade_Exposure,
                'Hedge Sensitivity': Hedge_Exposure,
                f'Factor Hedge Ratio (k_factor)': k_factor,
                'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct # MODIFIED: Column name update
            })
            
        except Exception as e:
            continue

    if not results:
        return pd.DataFrame(), "No valid hedge candidates found."
        
    results_df = pd.DataFrame(results)
    
    # Sort by Residual Volatility (Rate %) to show the most effective hedges first
    results_df = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True, na_position='last')
    
    return results_df, None

# --- NEW HELPER FUNCTION for Mispricing ---
def calculate_derivative_mispricings(historical_derivatives_list, analysis_dt):
    """
    Calculates the mispricing (Original - PCA Fair) in Rate % for all derivatives 
    on the analysis date. (Was BPS, now divided by 100)
    
    Args:
        historical_derivatives_list (list[pd.DataFrame]): List of all historical derivative DFs 
                                                         (containing 'Original' and 'PCA' columns).
        analysis_dt (datetime.datetime): The single analysis date for the snapshot.

    Returns:
        pd.Series: Series indexed by derivative label (without suffix), with mispricing in Rate % as values.
    """
    mispricing_data = {}
    
    # Ensure analysis_dt is aligned to the dataframe index (usually date component or string format)
    analysis_date_key = analysis_dt.strftime('%Y-%m-%d')
    
    for df in historical_derivatives_list:
        if df.empty or analysis_date_key not in df.index:
            continue
            
        try:
            # Try to get the row by the string key (works for DatetimeIndex)
            row = df.loc[analysis_date_key]
        except KeyError:
            continue
        
        # Iterate through all derivative columns that contain the original value
        for original_col in [col for col in df.columns if ' (Original)' in col]:
            pca_col = original_col.replace(' (Original)', ' (PCA)')
            
            if pca_col in row and not pd.isna(row[original_col]) and not pd.isna(row[pca_col]):
                # Remove the suffix to get the clean derivative label (e.g., '3M Spread: Z25-H26')
                derivative_label = original_col.replace(' (Original)', '')
                
                # Calculate mispricing in Rate %: (Original - PCA Fair) * 100 
                # MODIFIED: * 10000 -> * 100
                mispricing = (row[original_col] - row[pca_col]) * 100
                mispricing_data[derivative_label] = mispricing
                
    return pd.Series(mispricing_data, name='Hedge Mispricing (Rate %)') # MODIFIED: Column name update
# --- END NEW HELPER FUNCTION ---


# --- NEW FUNCTION FOR SECTION 8.3 ---
def create_instrument_universe_table(factor_sensitivities_df, Sigma_Raw_df, mispricing_series):
    """
    Creates a comprehensive table of all derivative instruments with their key attributes:
    Sensitivities, Total Volatility, and Mispricing.
    """
    if Sigma_Raw_df.empty or factor_sensitivities_df.empty:
        return pd.DataFrame()

    data = []
    
    # Calculate Total Volatility (Standard Deviation * 100)
    # Total Volatility is sqrt(Variance) * 100
    total_volatility = np.sqrt(np.diag(Sigma_Raw_df)) * 100
    total_vol_series = pd.Series(total_volatility, index=Sigma_Raw_df.index)

    for instrument in Sigma_Raw_df.index:
        
        # Determine Derivative Group (Spread, Fly, Double Fly)
        if 'Spread' in instrument and 'Double' not in instrument:
            instr_group = 'Spread'
        elif 'Double Fly' in instrument:
            instr_group = 'Double Fly'
        elif 'Fly' in instrument:
            instr_group = 'Fly'
        else:
            instr_group = 'Other'
            
        # Determine Maturity
        if '3M' in instrument:
            maturity = '3M'
        elif '6M' in instrument:
            maturity = '6M'
        elif '12M' in instrument:
            maturity = '12M'
        else:
            maturity = ''
            
        # Full Type
        full_type = f"{maturity} {instr_group}" if maturity else instr_group
        
        # Sensitivities (Handle missing factors if pc_count < 3)
        if instrument in factor_sensitivities_df.index:
            sensitivities = factor_sensitivities_df.loc[instrument]
            level_sens = sensitivities.get('Level (Whole Curve Shift)', np.nan)
            slope_sens = sensitivities.get('Slope (Steepening/Flattening)', np.nan)
            curve_sens = sensitivities.get('Curvature (Fly Risk)', np.nan)
        else:
            level_sens, slope_sens, curve_sens = np.nan, np.nan, np.nan
        
        # Mispricing (Rate %)
        mispricing = mispricing_series.get(instrument, np.nan)

        data.append({
            'Instrument': instrument,
            'Type': full_type,
            'Derivative Group': instr_group, # Column for filtering
            'Level Sensitivity': level_sens,
            'Slope Sensitivity': slope_sens,
            'Curvature Sensitivity': curve_sens,
            'Total Volatility (Rate %)': total_vol_series.loc[instrument],
            'Mispricing (Rate %)': mispricing
        })

    df = pd.DataFrame(data)
    return df
# --- END NEW FUNCTION ---


# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (e.g., 'SOFR rates.csv')", 
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

# Placeholder for L_D Loadings and Sigma_Raw_df, calculated in Section 7 and used in Section 8
loadings_df_gen = pd.DataFrame()
Sigma_Raw_df = pd.DataFrame()
spreads_3M_df_no_prefix = pd.DataFrame() # Also need this for Section 6 if price_df is not None

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
    
    analysis_dt = datetime.combine(analysis_date, datetime.min.time())

else:
    st.info("Please upload both the Price Data and Expiry Data CSV files to begin the analysis.")
    st.stop()


# --- Core Processing Logic ---
if not price_df_filtered.empty:
    
    # 1. Get the list of relevant contracts
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
    
    if future_expiries_df.empty:
        st.warning("Could not establish a relevant contract curve. Please check your date filters.")
        st.stop()

    # 2. Transform historical prices to the required maturity curve
    analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)
    
    if analysis_curve_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Derivatives
    st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    
    # 3M (k=1) - Used for PCA input
    spreads_3M_df_raw = calculate_k_step_spreads(analysis_curve_df, 1) # No prefix here
    butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
    double_butterflies_3M_df = calculate_k_step_double_butterflies(analysis_curve_df, 1) 
    
    # 6M (k=2)
    spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
    butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
    double_butterflies_6M_df = calculate_k_step_double_butterflies(analysis_curve_df, 2) 
    
    # 12M (k=4)
    spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
    butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
    double_butterflies_12M_df = calculate_k_step_double_butterflies(analysis_curve_df, 4) 
    
    
    # Display the number of contracts and derivatives
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Contracts", len(contract_labels))
    col2.metric("3M Spreads", spreads_3M_df_raw.shape[1])
    col3.metric("3M Flies", butterflies_3M_df.shape[1])
    col4.metric("3M Double Flies", double_butterflies_3M_df.shape[1])
    col5.metric("Date Range Days", price_df_filtered.shape[0])


    # 4. Perform PCA on 3M Spreads (Fair Curve)
    st.header("2. Principal Component Analysis (PCA) on 3M Spreads")
    
    loadings_spread, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df_raw)
    
    if loadings_spread is not None:
        
        # --- Variance Explained (Section 2) ---
        variance_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
            'Explained Variance (%)': explained_variance_ratio * 100
        })
        variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
        col_var, col_pca_select = st.columns([1, 1])
        
        with col_var:
            st.dataframe(variance_df, use_container_width=True)
            
        default_pc_count = min(3, len(explained_variance_ratio))
        
        with col_pca_select:
            st.subheader("Fair Curve & Hedging Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve & Hedging:", 
                min_value=1, 
                max_value=len(explained_variance_ratio), 
                value=default_pc_count,
                key='pc_slider'
            )
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads. This is the risk model used.")


        # --- Component Loadings Heatmaps (Section 3) ---
        st.header("3. PC Loadings")
        
        # --- 3.1 Spread Loadings (Standard Method) ---
        st.subheader("3.1 PC Loadings Heatmap (PC vs. 3M Spreads)")
        st.markdown("""
        This heatmap shows the **Loadings (Eigenvectors)** of the first few PCs on each **3-Month Spread**. These weights are derived from **Standardized PCA** and represent how each spread contributes to the overall risk factors (Level, Slope, Curvature).
        * **Interpretation of Loadings (Weights):** The value of the loading (weight) indicates the **sensitivity** of that specific spread to the respective Principal Component. A high absolute value means the spread has historically been highly correlated with the movement of that PC factor.
        """)
        
        plt.style.use('default')
        fig_spread_loading, ax_spread_loading = plt.subplots(figsize=(12, 6))
        
        # Only plot the first `default_pc_count` PCs in the heatmap
        loadings_spread_plot = loadings_spread.iloc[:, :default_pc_count]
        
        sns.heatmap(
            loadings_spread_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=0.5, 
            linecolor='gray', 
            cbar_kws={'label': 'Loading Weight'}
        )
        ax_spread_loading.set_title(f'3.1 Component Loadings for First {default_pc_count} Principal Components (on Spreads)', fontsize=16)
        ax_spread_loading.set_xlabel('Principal Component')
        ax_spread_loading.set_ylabel('Spread Contract')
        st.pyplot(fig_spread_loading)

        
        # --- 3.2 Outright Loadings (User Requested Non-Uniform PC1) ---
        st.subheader("3.2 Outright Price Loadings (Non-Uniform PC1)")
        st.markdown("""
        This heatmap is derived from **PCA on Outright Prices (Covariance Matrix)**, not the 3M spreads.
        The purpose is to show the raw, unstandardized **price sensitivity** of each contract to the first few PCs. This often results in a **Non-Uniform Level (PC1)** factor, which can be useful for visualizing the raw change in the entire curve.
        """)
        
        loadings_prices, explained_variance_prices = perform_pca_on_prices(analysis_curve_df)
        
        if loadings_prices is not None:
            
            fig_price_loading, ax_price_loading = plt.subplots(figsize=(12, 6))
            
            loadings_price_plot = loadings_prices.iloc[:, :default_pc_count]
            
            sns.heatmap(
                loadings_price_plot, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                linecolor='gray', 
                cbar_kws={'label': 'Loading Weight (Price Sensitivity)'}
            )
            ax_price_loading.set_title(f'3.2 Component Loadings for First {default_pc_count} Principal Components (on Outright Prices - Non-Uniform PC1)', fontsize=16)
            ax_price_loading.set_xlabel('Principal Component')
            ax_price_loading.set_ylabel('Contract')
            st.pyplot(fig_price_loading)
        else:
            st.warning("Outright Price PCA failed. Not enough contracts or data available.")
            
            
        # --- PC Factor Scores Time Series (Section 4) ---
        def plot_pc_scores(scores_df, explained_variance_ratio):
            """Plots the time series of the first 3 PC scores."""
            pc_labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
            num_pcs = min(3, scores_df.shape[1])
            
            if num_pcs == 0:
                return None
                
            fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
            if num_pcs == 1:
                axes = [axes]
                
            plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)
            
            for i in range(num_pcs):
                ax = axes[i]
                pc_label = pc_labels[i]
                variance_pct = explained_variance_ratio[i] * 100
                
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
        st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range. The scores are derived from the **Spread PCA (3.1)**.")
        
        fig_scores = plot_pc_scores(scores, explained_variance_ratio)
        if fig_scores:
            st.pyplot(fig_scores)


        # --- Historical Reconstruction (Based on Spread PCA) ---

        # 1. Reconstruct 3M Spreads using only selected PCs
        data_mean = spreads_3M_df_clean.mean()
        data_std = spreads_3M_df_clean.std()
        
        scores_used = scores.values[:, :pc_count]
        loadings_used = loadings_spread.values[:, :pc_count]
        
        # Inverse transform (Scores @ Loadings^T) * StdDev + Mean
        reconstructed_scaled = scores_used @ loadings_used.T
        
        reconstructed_spreads_3M = pd.DataFrame(
            reconstructed_scaled * data_std.values + data_mean.values,
            index=spreads_3M_df_clean.index,
            columns=spreads_3M_df_clean.columns
        )

        # 2. Reconstruct Outright Prices and ALL Derivatives (3M, 6M, 12M)
        historical_outrights_df, historical_spreads_3M_df, historical_butterflies_3M_df, historical_spreads_6M_df, historical_butterflies_6M_df, historical_spreads_12M_df, historical_butterflies_12M_df, historical_double_butterflies_3M_df, historical_double_butterflies_6M_df, historical_double_butterflies_12M_df, spreads_3M_df_no_prefix = reconstruct_prices_and_derivatives(
            analysis_curve_df, 
            reconstructed_spreads_3M, 
            spreads_3M_df_raw, 
            spreads_6M_df, 
            butterflies_3M_df, 
            butterflies_6M_df, 
            spreads_12M_df, 
            butterflies_12M_df,
            double_butterflies_3M_df, 
            double_butterflies_6M_df, 
            double_butterflies_12M_df
        )

        # --------------------------- Mispricing Calculation for Section 8 ---------------------------
        # Combine all historical derivative DFs (those containing Original and PCA columns)
        all_historical_derivatives_list = [
            historical_spreads_3M_df, historical_butterflies_3M_df, historical_double_butterflies_3M_df,
            historical_spreads_6M_df, historical_butterflies_6M_df, historical_double_butterflies_6M_df,
            historical_spreads_12M_df, historical_butterflies_12M_df, historical_double_butterflies_12M_df,
        ]
        
        mispricing_series = calculate_derivative_mispricings(all_historical_derivatives_list, analysis_dt)
        # --------------------------------------------------------------------------------------------------


        # --- Curve Snapshot (Section 5) ---
        st.header("5. Curve Snapshot (Original vs. PCA Fair Value)")
        

        def get_previous_date(df, current_date):
            """Return the last available previous date in df before current_date."""
            try:
                prev_dates = df.index[df.index < current_date]
                if len(prev_dates) == 0:
                    return None
                return prev_dates.max()
            except Exception:
                return None


        def plot_snapshot(historical_df, derivative_type, current_date, pc_count):
            """Plots the market vs PCA fair value snapshot (today vs previous day)."""

            try:
                # 1. Today's snapshot
                market_values = historical_df.loc[current_date].filter(like='(Original)')
                pca_fair_values = historical_df.loc[current_date].filter(like='(PCA)')

                # 2. Align and merge for plotting (today)
                comparison = pd.DataFrame({
                    'Original': market_values.values,
                    'PCA Fair': pca_fair_values.values
                }, index=[col.replace(f' (Original)', '').replace(f'{derivative_type}: ', '') for col in market_values.index])

                if comparison.empty:
                    st.info(f"No {derivative_type} data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                    return

                # 3. Previous-day snapshot
                prev_date = get_previous_date(historical_df, current_date)
                prev_series = None
                if prev_date is not None:
                    try:
                        prev_market = historical_df.loc[prev_date].filter(like='(Original)')
                        prev_series = pd.Series(
                            prev_market.values,
                            index=[col.replace(f' (Original)', '').replace(f'{derivative_type}: ', '') for col in prev_market.index],
                            name='Prev Day'
                        )
                    except KeyError:
                        prev_series = None

                # --- Plot the Derivative ---
                fig, ax = plt.subplots(figsize=(15, 7))

                ax.plot(
                    comparison.index,
                    comparison['Original'],
                    label=f'vwap',
                    marker='o',
                    linestyle='-',
                    linewidth=2.5,
                    color='blue'
                )
                ax.plot(
                    comparison.index,
                    comparison['PCA Fair'],
                    label=f'PCA',
                    marker='x',
                    linestyle='--',
                    linewidth=2.5,
                    color='red'
                )

                # Previous-day original curve, if available
                if prev_series is not None:
                    ax.plot(
                        prev_series.index,
                        prev_series.values,
                        label=f'settle',
                        marker='s',
                        linestyle='-.',
                        linewidth=2.0,
                        color='green'
                    )

                mispricing = comparison['Original'] - comparison['PCA Fair']
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

                # Annotate the derivative with the largest absolute mispricing (today)
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 100  # Rate %

                    ax.annotate(
                        f"Mispricing: {mispricing_value:.4f} Rate %",
                        (mispricing_contract, comparison.loc[mispricing_contract]['Original']), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )

                ax.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type} (Today vs Prev Day)', fontsize=16)
                ax.set_xlabel(f'{derivative_type} Contract')
                ax.set_ylabel(f'{derivative_type} Value (Price Difference)')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True, linestyle=':', alpha=0.6)

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

                # Collect Section 5 figure for PDF download
                SECTION5_FIGURES.append((fig, f"Section 5 – {derivative_type}"))

                # --- Detailed Table ---
                st.markdown(f"###### {derivative_type} Mispricing (Today vs PCA, with Prev Day if available)")
                detailed_comparison = comparison.copy()
                detailed_comparison.index.name = f'{derivative_type} Contract'
                detailed_comparison['Mispricing (Rate %)'] = mispricing * 100 
                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': f'Original {derivative_type}', 'PCA Fair': f'PCA Fair {derivative_type}'}
                )

                # Add previous-day original column if exists
                if prev_series is not None:
                    prev_align = prev_series.reindex(detailed_comparison.index)
                    detailed_comparison[f'Prev Day Original {derivative_type} ({prev_date.strftime("%Y-%m-%d")})'] = prev_align.values

                st.dataframe(
                    detailed_comparison.style.format({
                        f'Original {derivative_type}': "{:.4f}",
                        f'PCA Fair {derivative_type}': "{:.4f}",
                        'Mispricing (Rate %)': "{:.4f}"
                    }), 
                    use_container_width=True
                )

            except KeyError:
                 st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for {derivative_type}. Please choose a different date within the historical range.")
        def plot_shock_derivative_snapshot(historical_df, derivative_type, shocked_series, current_date, pc_count, title_suffix=""):
            """
            Plots Original vs PCA Fair vs Shock Scenario for a given derivative family
            on the selected analysis date, using the same x-axis ordering as Section 5.
            """
            try:
                row = historical_df.loc[current_date]
            except KeyError:
                st.info(f"No {derivative_type} data available for the selected analysis date in shock snapshot.")
                return

            market_values = row.filter(like='(Original)')
            pca_fair_values = row.filter(like='(PCA)')

            if market_values.empty or pca_fair_values.empty:
                st.info(f"{derivative_type}: Missing Original or PCA Fair values for shock snapshot.")
                return

            # Build a clean instrument index WITHOUT tenor prefixes (e.g. '3M Spread: ')
            base_index = []
            for col in market_values.index:
                core = col.replace(' (Original)', '')
                if ': ' in core:
                    core = core.split(': ', 1)[1]
                base_index.append(core)

            comparison = pd.DataFrame(
                {
                    'Original': market_values.values,
                    'PCA Fair': pca_fair_values.values,
                },
                index=base_index,
            )

            if shocked_series is None or len(shocked_series) == 0:
                st.info(f"No shocked series supplied for {derivative_type} in shock snapshot.")
                return

            shocked_aligned = shocked_series.reindex(comparison.index)
            if shocked_aligned.isna().all():
                st.info(f"Shocked series for {derivative_type} could not be aligned to instruments.")
                return

            comparison['Shock Scenario'] = shocked_aligned.values

            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(comparison.index, comparison['Original'], label=f'{derivative_type} Original', marker='o')
            ax.plot(comparison.index, comparison['PCA Fair'], label=f'{derivative_type} PCA Fair ({pc_count} PCs)', marker='x', linestyle='--')
            ax.plot(comparison.index, comparison['Shock Scenario'], label=f'{derivative_type} Shock {title_suffix}', marker='s', linestyle='-.')

            ax.set_title(f'{derivative_type} Snapshot under Shock {title_suffix}')
            ax.set_xlabel('Instrument')
            ax.set_ylabel('Value (Price Points)')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            # Collect Section 9 shock figure for PDF download
            SECTION9_FIGURES.append((fig, f"Section 9 – {derivative_type} {title_suffix}".strip()))


        # --- 5.1 Outright Price/Rate Curve Snapshot ---

        st.subheader("5.1 Outright Price/Rate Curve Snapshot")
        try:
            # 1. Get the snapshot for the selected date
            market_prices = historical_outrights_df.loc[analysis_dt].filter(like='(Original)')
            pca_fair_prices = historical_outrights_df.loc[analysis_dt].filter(like='(PCA)')

            # 2. Align and merge for plotting
            curve_comparison = pd.DataFrame({
                'Original': market_prices.values,
                'PCA Fair': pca_fair_prices.values
            }, index=[col.replace(' (Original)', '') for col in market_prices.index])

            # --- Plot the Curve (Today vs Previous Day) ---
            fig_curve, ax_curve = plt.subplots(figsize=(15, 7))

            # Today
            ax_curve.plot(
                curve_comparison.index,
                curve_comparison['Original'],
                label=f'Today Original Price ({analysis_dt.strftime("%Y-%m-%d")})',
                marker='o',
                linestyle='-',
                linewidth=2.5,
                color='blue'
            )
            ax_curve.plot(
                curve_comparison.index,
                curve_comparison['PCA Fair'],
                label=f'Today PCA Fair Price ({pc_count} PCs)',
                marker='x',
                linestyle='--',
                linewidth=2.5,
                color='red'
            )

            # Previous day
            prev_dt = get_previous_date(historical_outrights_df, analysis_dt)
            if prev_dt is not None:
                try:
                    prev_prices = historical_outrights_df.loc[prev_dt].filter(like='(Original)')
                    prev_cmp = pd.Series(
                        prev_prices.values,
                        index=[col.replace(' (Original)', '') for col in prev_prices.index]
                    )
                    ax_curve.plot(
                        prev_cmp.index,
                        prev_cmp.values,
                        label=f'Prev Day Original Price ({prev_dt.strftime("%Y-%m-%d")})',
                        marker='s',
                        linestyle='-.',
                        linewidth=2.0,
                        color='green'
                    )
                except KeyError:
                    pass

            ax_curve.set_title('Market Price Curve vs. PCA Fair Value Curve (Price = 100 - Rate, Today vs Prev Day)', fontsize=16)
            ax_curve.set_xlabel('Contract Maturity')
            ax_curve.set_ylabel('Price (100 - Rate)')
            ax_curve.legend(loc='upper right')
            ax_curve.grid(True, linestyle=':', alpha=0.6)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_curve)

            # Collect Outright curve figure for Section 5 PDF
            SECTION5_FIGURES.append((fig_curve, "Section 5 – Outright Curve"))

            # --- Detailed Contract Price/Rate Table (Outright) ---
            st.markdown("###### Outright Price and Rate Mispricing")
            detailed_comparison = curve_comparison.copy()
            detailed_comparison.index.name = 'Contract'
            detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
            detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
            detailed_comparison['Mispricing (Rate %)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 100

            detailed_comparison = detailed_comparison.rename(
                columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'}
            )
            detailed_comparison = detailed_comparison[[
                'Original Price', 'Original Rate (%)', 'PCA Fair Price', 'PCA Fair Rate (%)', 'Mispricing (Rate %)'
            ]]

            st.dataframe(
                detailed_comparison.style.format({
                    'Original Price': "{:.4f}",
                    'PCA Fair Price': "{:.4f}",
                    'Original Rate (%)': "{:.4f}",
                    'PCA Fair Rate (%)': "{:.4f}",
                    'Mispricing (Rate %)': "{:.4f}"
                }), 
                use_container_width=True
            )
        except KeyError:
            st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Outright Prices. Please choose a different date within the historical range.")
        # --------------------------- 3-Month (k=1) Derivatives ---------------------------
        # --- 5.2 Spread Snapshot (3M) ---
        st.subheader("5.2 3M Spread Snapshot (k=1, e.g., Z25-H26)")
        plot_snapshot(historical_spreads_3M_df, "3M Spread", analysis_dt, pc_count)

        # --- 5.3 Butterfly (Fly) Snapshot (3M) ---
        if not historical_butterflies_3M_df.empty:
            st.subheader("5.3 3M Butterfly (Fly) Snapshot (k=1, e.g., Z25-2xH26+M26)")
            plot_snapshot(historical_butterflies_3M_df, "3M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 3 or more) to calculate and plot 3M butterfly snapshot.")
            
        # --- 5.4 Double Butterfly (DBF) Snapshot (3M) --- 
        if not historical_double_butterflies_3M_df.empty:
            st.subheader(r"5.4 3M Double Butterfly (DBF) Snapshot ($k=1$, e.g., $Z25-3 \cdot H26+3 \cdot M26-U26$)")
            plot_snapshot(historical_double_butterflies_3M_df, "3M Double Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 4 or more) to calculate and plot 3M double butterfly snapshot.")
            
        # --------------------------- 6-Month (k=2) Derivatives ---------------------------
        # --- 5.5 Spread Snapshot (6M) ---
        st.subheader("5.5 6M Spread Snapshot (k=2, e.g., Z25-M26)")
        plot_snapshot(historical_spreads_6M_df, "6M Spread", analysis_dt, pc_count)
        
        # --- 5.6 Butterfly (Fly) Snapshot (6M) ---
        if not historical_butterflies_6M_df.empty:
            st.subheader("5.6 6M Butterfly (Fly) Snapshot (k=2, e.g., Z25-2xM26+Z26)")
            plot_snapshot(historical_butterflies_6M_df, "6M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 5 or more) to calculate and plot 6M butterfly snapshot.")
            
        # --- 5.7 Double Butterfly (DBF) Snapshot (6M) --- 
        if not historical_double_butterflies_6M_df.empty:
            st.subheader(r"5.7 6M Double Butterfly (DBF) Snapshot ($k=2$, e.g., $Z25-3 \cdot M26+3 \cdot Z26-M27$)")
            plot_snapshot(historical_double_butterflies_6M_df, "6M Double Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 7 or more) to calculate and plot 6M double butterfly snapshot.")

        # --------------------------- 12-Month (k=4) Derivatives ---------------------------
        # --- 5.8 Spread Snapshot (12M) ---
        st.subheader("5.8 12M Spread Snapshot (k=4, e.g., Z25-Z26)")
        plot_snapshot(historical_spreads_12M_df, "12M Spread", analysis_dt, pc_count)

        # --- 5.9 Butterfly (Fly) Snapshot (12M) ---
        if not historical_butterflies_12M_df.empty:
            st.subheader("5.9 12M Butterfly (Fly) Snapshot (k=4, e.g., Z25-2xZ26+Z27)")
            plot_snapshot(historical_butterflies_12M_df, "12M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 9 or more) to calculate and plot 12M butterfly snapshot.")

        # --- 5.10 Double Butterfly (DBF) Snapshot (12M) --- 
        if not historical_double_butterflies_12M_df.empty:
            st.subheader(r"5.10 12M Double Butterfly (DBF) Snapshot ($k=4$, e.g., $Z25-3 \cdot Z26+3 \cdot Z27-Z28$)")
            plot_snapshot(historical_double_butterflies_12M_df, "12M Double Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 13 or more) to calculate and plot 12M double butterfly snapshot.")

        
        # --------------------------- Download all Section 5 snapshots as PDF ---------------------------
        st.subheader("Download All Section 5 Snapshots as PDF")

        if not SECTION5_FIGURES:
            st.info("Generate the Section 5 charts above to enable PDF download.")
        else:
            pdf_buffer_5 = BytesIO()
            with PdfPages(pdf_buffer_5) as pdf:
                for fig, title in SECTION5_FIGURES:
                    if title:
                        fig.suptitle(title)
                    pdf.savefig(fig, bbox_inches="tight")

            pdf_buffer_5.seek(0)

            st.download_button(
                label="📥 Download Section 5 Snapshots as PDF",
                data=pdf_buffer_5,
                file_name="SOFR.pdf",
                mime="application/pdf",
            )

        # --------------------------- 6. PCA-Based Hedging Strategy (3M Spreads ONLY - Original Section) ---------------------------
        st.header("6. PCA-Based Hedging Strategy (3M Spreads ONLY - Original Section)")
        # FIX: The following text must be wrapped in st.markdown() to prevent NameError
        st.markdown(f"""
        This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for a chosen **3M spread** trade, using *another 3M spread* as the hedge. The calculation uses the **Covariance Matrix** of the **3M spreads**, which is **reconstructed using the selected {pc_count} Principal Components**.
        * **Trade:** Long 1 unit of the selected 3M spread.
        * **Hedge:** Short $k^*$ units of the hedging 3M spread.
        * **Volatility:** Expressed as **Rate %** ($1\% = 100 \text{{ BPS}}$).
        """) 
        
        if spreads_3M_df_clean.shape[1] < 2:
            st.warning("Not enough 3M spreads available to calculate a hedge.")
        else:
            # Drop the prefixes for this section since the function is designed for 3M spreads without prefixes
            spread_labels_3m = spreads_3M_df_no_prefix.columns.tolist()
            trade_selection_3m = st.selectbox(
                "Select 3M Spread Trade Instrument (T)", 
                options=spread_labels_3m,
                key='trade_3m_select'
            )
            
            # Run the hedging analysis
            best_hedge_data_3m, worst_hedge_data_3m, all_results_df_full_3m = calculate_best_and_worst_hedge_3M(
                trade_selection_3m, loadings_spread, eigenvalues, pc_count, spreads_3M_df_clean
            )
            
            if best_hedge_data_3m is not None:
                st.subheader(f"Trade: Long 1 unit of **{trade_selection_3m}**")
                
                # --- Best Hedge ---
                st.markdown("#### Best Hedge (Minimum Residual Risk)")
                st.markdown(f"""
                - **Hedge Instrument (H):** **{best_hedge_data_3m['Hedge Spread']}**
                - **Hedge Action:** Short **{best_hedge_data_3m['Hedge Ratio (k*)']:.4f}** units.
                - **Residual Volatility (Rate %):** **{best_hedge_data_3m['Residual Volatility (Rate %)']:.4f} Rate %** (Lowest Risk) # MODIFIED: Name and format update
                """)
                
                # --- Worst Hedge ---
                st.markdown("#### Worst Hedge (Maximum Residual Risk)")
                st.markdown(f"""
                - **Hedge Instrument (H):** **{worst_hedge_data_3m['Hedge Spread']}**
                - **Hedge Action:** Short **{worst_hedge_data_3m['Hedge Ratio (k*)']:.4f}** units.
                - **Residual Volatility (Rate %):** **{worst_hedge_data_3m['Residual Volatility (Rate %)']:.4f} Rate %** (Highest Risk) # MODIFIED: Name and format update
                """)
                
                st.markdown("---")
                st.markdown("###### Detailed Hedging Results (All 3M Spreads as Hedge Candidates - Sorted by Minimum Variance)")
                # Use the full results DataFrame directly and sort it for display
                all_results_df_full_3m = all_results_df_full_3m.sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                st.dataframe(
                    all_results_df_full_3m.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (Rate %)': "{:.4f}" # MODIFIED: Name and format update
                    }), 
                    use_container_width=True
                )
            else:
                st.warning("3M Hedging calculation failed. Check if enough historical data is available after filtering.")


        # --------------------------- 7. PCA-Based Generalized Hedging Strategy (Minimum Variance) ---------------------------
        st.header("7. PCA-Based Generalized Hedging Strategy (Minimum Variance)")
        st.markdown(f"""
        This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for *any* derivative trade, using *any* other derivative as a hedge. The calculation is based on the **full covariance matrix** of all derivatives, which is **reconstructed using the selected {pc_count} Principal Components** derived from the 3M Spreads.
        * **Trade:** Long 1 unit of the selected instrument.
        * **Hedge:** Short $k^*$ units of the hedging instrument.
        * **Volatility:** Expressed as **Rate %** ($1\% = 100 \text{{ BPS}}$).
        """) # MODIFIED: Note on Rate % update

        # --- HEDGING DATA PREPARATION (FOR SECTIONS 7 & 8) ---
        
        # 1. Combine all historical derivative time series into one DataFrame
        # **CRITICAL: Ensure all derivatives have unique, explicit prefixes**
        all_derivatives_list = [
            spreads_3M_df_raw.rename(columns=lambda x: f"3M Spread: {x}"), # Uses raw spread DF (no prefix)
            butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"),
            double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"), 
            
            spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"),
            butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"),
            double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}"), 
            
            spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}"),
            butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}"),
            double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}") 
        ]
        
        # Only keep non-empty dataframes
        all_derivatives_df_raw = pd.concat([df for df in all_derivatives_list if not df.empty], axis=1)

        # 2. Calculate Generalized Covariance Matrix (Sigma_Raw_df) and Loadings (loadings_df_gen)
        Sigma_Raw_df, all_derivatives_df_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(
            all_derivatives_df_raw, scores, eigenvalues, pc_count
        )
        
        
        if not Sigma_Raw_df.empty and Sigma_Raw_df.shape[1] > 1:
            
            # Get the list of all available derivative instruments
            all_derivatives_labels = Sigma_Raw_df.columns.tolist()
            
            trade_selection_gen = st.selectbox(
                "Select Trade Instrument (T)", 
                options=all_derivatives_labels,
                key='trade_gen_select'
            )
            
            # Run the generalized hedging analysis
            best_hedge_data_gen, worst_hedge_data_gen, all_results_df_full_gen = calculate_best_and_worst_hedge_generalized(
                trade_selection_gen, Sigma_Raw_df
            )
            
            if best_hedge_data_gen is not None:
                st.subheader(f"Trade: Long 1 unit of **{trade_selection_gen}**")
                
                # --- Best Hedge ---
                st.markdown("#### Best Hedge (Minimum Residual Risk)")
                st.markdown(f"""
                - **Hedge Instrument (H):** **{best_hedge_data_gen['Hedge Instrument']}**
                - **Hedge Action:** Short **{best_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                - **Residual Volatility (Rate %):** **{best_hedge_data_gen['Residual Volatility (Rate %)']:.4f} Rate %** (Lowest Risk) # MODIFIED: Name and format update
                """)
                
                # --- Worst Hedge ---
                st.markdown("#### Worst Hedge (Maximum Residual Risk)")
                st.markdown(f"""
                - **Hedge Instrument (H):** **{worst_hedge_data_gen['Hedge Instrument']}**
                - **Hedge Action:** Short **{worst_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                - **Residual Volatility (Rate %):** **{worst_hedge_data_gen['Residual Volatility (Rate %)']:.4f} Rate %** (Highest Risk) # MODIFIED: Name and format update
                """)
                
                st.markdown("---")
                st.markdown("###### Detailed Hedging Results (All Derivatives as Hedge Candidates - Sorted by Minimum Variance)")
                # Use the full results DataFrame directly and sort it for display
                all_results_df_full_gen = all_results_df_full_gen.sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                st.dataframe(
                    all_results_df_full_gen.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (Rate %)': "{:.4f}" # MODIFIED: Name and format update
                    }), 
                    use_container_width=True
                )
            else:
                st.warning("Generalized Minimum Variance Hedging calculation failed for the selected trade. Check if enough historical data is available after filtering.")


        # --------------------------- 8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging - MODIFIED) ---------------------------
        st.header("8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging)")
        st.markdown(f"""
        This strategy uses the Level, Slope, and Curvature factors (PC1, PC2, PC3) to identify hedges that neutralize specific factor exposures.
        * **Factor Exposures:** Standardized sensitivities (Beta) to the principal components.
        * **Volatility/Mispricing:** Expressed as **Rate %** ($1\% = 100 \text{{ BPS}}$).
        """) 
        
        # 1. Calculate Factor Sensitivities (L_D columns renamed)
        factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count)
        
        if not factor_sensitivities_df.empty and not Sigma_Raw_df.empty:
            
            # --- User Selections ---
            all_derivatives_labels_factor = factor_sensitivities_df.index.tolist()
            factor_names = factor_sensitivities_df.columns.tolist()
            
            col_trade_select, col_factor_select = st.columns(2)
            
            with col_trade_select:
                trade_selection_factor = st.selectbox(
                    "Select Trade Instrument (T)", 
                    options=all_derivatives_labels_factor,
                    key='trade_factor_select'
                )
                
            with col_factor_select:
                st.info("Results will display the best hedge for all factors.")

            st.markdown("---")

            # --- 8.1 NEW: Triple Factor Neutralization Check ---
            st.subheader(f"8.1 **Triple Factor Neutralization** Check (Trade: {trade_selection_factor})")
            st.markdown(r"""
            This checks if any *single* hedge instrument **($H$)** can simultaneously neutralize the trade's **Level, Slope, and Curvature** exposure. This requires the ratio of factor sensitivities ($\frac{E_{PCi}(T)}{E_{PCi}(H)}$) to be nearly identical for all three factors, resulting in a single hedge ratio ($k$):
            $$\frac{E_{Level}(T)}{E_{Level}(H)} \approx \frac{E_{Slope}(T)}{E_{Slope}(H)} \approx \frac{E_{Curvature}(T)}{E_{Curvature}(H)} = k$$
            """)
            
            # Check for Triple Factor Hedge
            triple_hedge_check_result = find_perfect_factor_hedge(
                trade_selection_factor, 
                factor_sensitivities_df, 
                mispricing_series, 
                pc_count
            )
            
            if triple_hedge_check_result['result'] is not None:
                res = triple_hedge_check_result['result']
                
                # --- Display the results in a clear table ---
                triple_data = {
                    'Metric': [
                        'Trade Instrument', 
                        'Hedge Instrument (H)', 
                        'Hedge Action',
                        'Hedge Ratio (|k|)',
                        'Trade PC1 (Level) Sensitivity', 
                        'Hedge PC1 (Level) Sensitivity', 
                        'Trade PC2 (Slope) Sensitivity',
                        'Hedge PC2 (Slope) Sensitivity',
                        'Trade PC3 (Curvature) Sensitivity',
                        'Hedge PC3 (Curvature) Sensitivity',
                        'Hedge Mispricing (Rate %)',
                        'Max K Difference (Tolerance Check)'
                    ],
                    'Value': [
                        trade_selection_factor,
                        res['Hedge Instrument'],
                        f"{res['Hedge Action']} {res['Hedge Ratio (|k|)']:.4f} units",
                        f"{res['Hedge Ratio (|k|)']:.4f}",
                        f"{res['Trade PC1 Sensitivity']:.4f}",
                        f"{res['Hedge PC1 Sensitivity']:.4f}",
                        f"{res['Trade PC2 Sensitivity']:.4f}",
                        f"{res['Hedge PC2 Sensitivity']:.4f}",
                        f"{res['Trade PC3 Sensitivity']:.4f}",
                        f"{res['Hedge PC3 Sensitivity']:.4f}",
                        f"{res['Hedge Mispricing (Rate %)']:.4f}" if not np.isnan(res['Hedge Mispricing (Rate %)']) else 'N/A',
                        f"{res['Max K Difference']:.6e}"
                    ]
                }
                
                st.success(f"**PERFECT FACTOR HEDGE FOUND!** The instrument **{res['Hedge Instrument']}** can neutralize the first three factors simultaneously.")
                st.table(pd.DataFrame(triple_data).set_index('Metric'))
                
            else:
                st.info(triple_hedge_check_result['error'])

            st.markdown("---") 

            # --- 8.2 Single Factor Neutralization Results ---
            st.subheader(f"8.2 **Single Factor Neutralization** Results (Trade: {trade_selection_factor})")
            st.markdown(f"The best hedge for each single factor minimizes the total remaining (residual) risk after neutralizing that specific factor's exposure.")
            
            summary_results = []
            
            # --- Run Hedging Analysis for All Factors ---
            for target_factor in factor_names:
                factor_results_df, error_msg = calculate_all_factor_hedges(
                    trade_selection_factor, target_factor, factor_sensitivities_df, Sigma_Raw_df
                )
                
                if error_msg:
                    continue
                
                # Filter out hedges with near-zero factor sensitivity (Ratio is meaningless/too large)
                factor_results_df_clean = factor_results_df.dropna(subset=['Residual Volatility (Rate %)']) # MODIFIED: Column name update
                
                if not factor_results_df_clean.empty:
                    # Find the SINGLE best hedge (minimum residual volatility) for the current factor
                    best_hedge_row = factor_results_df_clean.iloc[0]
                    
                    # --- FETCH HEDGE MISPRICING ---
                    best_hedge_instrument = best_hedge_row['Hedge Instrument']
                    # Use .get() to safely retrieve mispricing, defaulting to NaN if not found
                    hedge_mispricing = mispricing_series.get(best_hedge_instrument, np.nan) 
                    # ----------------------------
                    
                    # Determine the Hedge Action (Short/Long) based on the Hedge Ratio
                    k_factor_value = best_hedge_row[f'Factor Hedge Ratio (k_factor)']
                    if k_factor_value > 0:
                        hedge_action = 'Short'
                    elif k_factor_value < 0:
                        hedge_action = 'Long'
                    else:
                        hedge_action = 'N/A' # Should be rare if k_factor is non-zero
                        
                    summary_results.append({
                        'Factor to Neutralize': target_factor,
                        'Hedge Instrument': best_hedge_row['Hedge Instrument'],
                        'Hedge Action': hedge_action,
                        'Hedge Ratio (|k|)': abs(k_factor_value),
                        'Residual Volatility (Rate %)': best_hedge_row['Residual Volatility (Rate %)'], # MODIFIED: Column name update
                        'Hedge Mispricing (Rate %)': hedge_mispricing, # MODIFIED: Column name update
                        'Trade Sensitivity': best_hedge_row['Trade Sensitivity'],
                        'Hedge Sensitivity': best_hedge_row['Hedge Sensitivity']
                    })

            # --- Display Summary Table of Best Factor Hedges ---
            if summary_results:
                summary_df = pd.DataFrame(summary_results).sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                # MODIFICATION: Insert 'Hedge Mispricing (BPS)' into the displayed columns
                st.dataframe(
                    summary_df[[
                        'Factor to Neutralize', 
                        'Hedge Instrument', 
                        'Hedge Action', 
                        'Hedge Ratio (|k|)', 
                        'Residual Volatility (Rate %)', # MODIFIED: Column name update
                        'Hedge Mispricing (Rate %)', # MODIFIED: Column name update
                        'Trade Sensitivity', 
                        'Hedge Sensitivity'
                    ]].style.format({
                        'Trade Sensitivity': "{:.4f}",
                        'Hedge Sensitivity': "{:.4f}",
                        'Hedge Ratio (|k|)': "{:.4f}",
                        'Residual Volatility (Rate %)': "{:.4f}", # MODIFIED: Format to 4 decimals for clarity
                        'Hedge Mispricing (Rate %)': "{:.4f}", # MODIFIED: Format to 4 decimals for clarity
                    }),
                    use_container_width=True
                )
                
                # --- NEW EXPLANATION OF THE TABLE ---
                st.markdown("---")
                st.markdown("### 💡 Explanation of Single Factor Hedging Results")
                st.markdown("""
                The table in **Section 8.2** shows the **ideal hedge instrument** to neutralize the risk from a *single, specific market factor* (Level, Slope, or Curvature).

                A hedge is considered 'better' in this context because it **minimizes the Residual Volatility** for that specific factor's risk:
                
                1.  **Factor Neutralization:** The `Factor Hedge Ratio (|k|)` is calculated as the ratio of the Trade's sensitivity to the Hedge's sensitivity for the target factor ($\frac{E_{Factor}(T)}{E_{Factor}(H)}$). When you enter the trade and the hedge at this ratio, the total portfolio exposure to that factor becomes zero.
                
                2.  **Minimum Residual Volatility:** While the factor risk is zeroed out, residual risk from **all other factors** remains. The instrument displayed is the one that achieves that **factor neutrality** while simultaneously resulting in the **lowest overall residual risk** (as measured by `Residual Volatility (Rate %)`). This is determined using the full covariance matrix (Section 7's $\Sigma_{Raw}$) to precisely calculate the remaining, unhedged volatility.

                3.  **Hedge Mispricing (Rate %):** This column provides the key trading signal. It shows the difference between the market price of the hedge instrument and its PCA Fair Value (`Original Price - PCA Fair Value`).
                    * **A high absolute mispricing** combined with a **low residual volatility** suggests a potentially **high-quality, high-alpha trade**. You are using an attractively mispriced instrument to neutralize a major risk factor, leaving only minimal idiosyncratic (unexplained) risk.
                """)
                # --- END NEW EXPLANATION ---
            
            else:
                 st.info(f"No valid factor hedge candidates found for trade **{trade_selection_factor}** across Level, Slope, or Curvature.")


            st.markdown("---") 

            # --- 8.3 Filtered Universe of Potential Hedges ---
            st.header("8.3 Filtered Universe of Potential Hedges")
            st.markdown("""
            This table provides a comprehensive view of all available derivative instruments, categorized by type (Spread, Fly, Double Fly). It presents the instrument's **risk attributes** (Sensitivities, Total Volatility) and its **trading signal** (Mispricing) to help identify high-quality hedging instruments.
            
            * **Note:** The hedging model is based on PCA of **Spreads/Derivatives**. Outright contracts are excluded here as they do not have the same standardized Level/Slope/Curvature factor exposures.
            """)
            
            # 1. Create the universe table
            instrument_universe_df = create_instrument_universe_table(factor_sensitivities_df, Sigma_Raw_df, mispricing_series)
            
            if not instrument_universe_df.empty:
                
                # 2. Add Filter
                derivative_options = ['All Derivatives'] + sorted(instrument_universe_df['Derivative Group'].unique().tolist())
                
                # Exclude 'Other' if it's the only option or empty
                if len(derivative_options) > 2 and 'Other' in derivative_options:
                    derivative_options.remove('Other')
                    
                selected_group = st.radio(
                    "Select Derivative Group to View:", 
                    options=derivative_options,
                    index=0,
                    key='derivative_filter_83',
                    horizontal=True
                )
                
                # 3. Filter the table
                if selected_group != 'All Derivatives':
                    filtered_df = instrument_universe_df[instrument_universe_df['Derivative Group'] == selected_group]
                else:
                    filtered_df = instrument_universe_df.copy()
                
                # 4. Prepare for display and sort
                display_df = filtered_df.drop(columns=['Derivative Group']).sort_values(
                    by='Total Volatility (Rate %)', 
                    ascending=False
                )
                
                # 5. Display the table
                st.markdown(f"###### Attributes for: **{selected_group}** (Total Instruments: {len(display_df)})")
                st.dataframe(
                    display_df.style.format({
                        'Level Sensitivity': "{:.4f}",
                        'Slope Sensitivity': "{:.4f}",
                        'Curvature Sensitivity': "{:.4f}",
                        'Total Volatility (Rate %)': "{:.4f}",
                        'Mispricing (Rate %)': "{:.4f}"
                    }).background_gradient(
                        subset=['Mispricing (Rate %)'], 
                        cmap='coolwarm', 
                        vmax=display_df['Mispricing (Rate %)'].abs().max() * 0.5 if not display_df['Mispricing (Rate %)'].abs().empty else 0.5,
                        vmin=-display_df['Mispricing (Rate %)'].abs().max() * 0.5 if not display_df['Mispricing (Rate %)'].abs().empty else -0.5 # Gradient strength
                    ),
                    use_container_width=True
                )
                
                st.markdown("""
                ### 🎯 How to use this table for hedging:
                * **Identify Mispriced Hedges (Signal):** Look for instruments with a high absolute **Mispricing (Rate %)** (deep red or deep blue in the background gradient). This is your potential *alpha* source.
                * **Assess Factor Exposure (Risk Match):** Check the **Level, Slope, and Curvature Sensitivity**. If your main trade is exposed to the Slope factor, you'll need a hedge with a strong, opposite Slope Sensitivity.
                * **Evaluate Hedge Impact (Risk):** The **Total Volatility (Rate %)** is the inherent risk of the hedge instrument itself. Using a high volatility hedge (top of the list) will require a more precise hedge ratio to avoid adding more risk than you remove.
                """)

            else:
                st.info("Instrument universe table could not be created. Ensure enough historical data is available.")

             
            # Display full sensitivities table as before for reference
            st.markdown("---")
            st.subheader(f"Factor Sensitivities (Standardized Beta) Table for Reference")
            st.markdown("This shows the raw input exposures used for the ratio calculation. Note: Outright prices are not included here as factor hedging applies to the derivatives used in the PCA structure.")
            
            st.dataframe(
                factor_sensitivities_df.style.format("{:.4f}"),
                use_container_width=True
            )

        

# --------------------------- 8.4 Historical Backtest of Trade + Hedge Pair ---------------------------
        st.markdown("---")
        st.subheader("8.4 Historical Backtest: Trade + Hedge Portfolio")

        st.markdown(r"""
        This section lets you **simulate the historical behaviour** of a **Trade + Hedge** combination:

        * You pick:
          - A **trade instrument** and direction/size.
          - A **hedge instrument** and hedge ratio $k$ (portfolio is $P = T - kH$).
        * The tool then:
          - Builds daily **P&L time series** for Trade, Hedge, and the combined portfolio.
          - Computes **volatility before vs after hedging**.
          - Shows the **cumulative P&L** evolution through time.

        This is exactly how a bank desk sanity-checks hedges before putting risk on.
        """)

        # --- Helper: safely retrieve historical price series for a derivative label ---
        def _get_price_series_for_label(derivative_label: str):
            """
            Safely retrieve the historical price series for a derivative label like:
            "3M Spread: Z25-Z26", "6M Fly: Z25-Z27", etc.

            This version:
              ✔ uses the *_df naming convention consistently
              ✔ checks globals() before accessing
              ✔ gracefully returns None if data is missing
            """
            if ":" not in derivative_label:
                return None

            prefix, rest = derivative_label.split(": ", 1)
            type_key = prefix.strip()

            # Map instrument family to the standard *_df historical dataframes
            hist_map_names = {
                "3M Spread": "historical_spreads_3M_df",
                "3M Fly": "historical_butterflies_3M_df",
                "3M Double Fly": "historical_double_butterflies_3M_df",

                "6M Spread": "historical_spreads_6M_df",
                "6M Fly": "historical_butterflies_6M_df",
                "6M Double Fly": "historical_double_butterflies_6M_df",

                "12M Spread": "historical_spreads_12M_df",
                "12M Fly": "historical_butterflies_12M_df",
                "12M Double Fly": "historical_double_butterflies_12M_df",
            }

            if type_key not in hist_map_names:
                return None

            dataset_name = hist_map_names[type_key]

            # Ensure the dataframe actually exists in the global namespace
            if dataset_name not in globals():
                return None

            df = globals()[dataset_name]
            if df is None or df.empty:
                return None

            col_name = f"{derivative_label} (Original)"
            if col_name not in df.columns:
                return None

            return df[col_name].dropna()

        def _compute_hedged_pnl_series(
            trade_label: str,
            hedge_label: str,
            trade_direction: str,
            trade_units: float,
            hedge_ratio_k: float
        ):
            """
            Build daily P&L for Trade, Hedge and Portfolio:
                P_T = sign_T * N_T * ΔT
                P_H = -k * N_T * ΔH
            Portfolio PnL = P_T + P_H

            This uses daily *differences* in the instrument prices (already spreads/flies).
            """

            trade_series = _get_price_series_for_label(trade_label)
            hedge_series = _get_price_series_for_label(hedge_label)

            if trade_series is None:
                st.error(f"Historical data not found for trade instrument: {trade_label}")
                return None
            if hedge_series is None:
                st.error(f"Historical data not found for hedge instrument: {hedge_label}")
                return None

            df_prices = pd.concat(
                [trade_series.rename("Trade"), hedge_series.rename("Hedge")],
                axis=1
            ).dropna()

            if df_prices.empty:
                st.error("No overlapping history between trade and hedge instruments.")
                return None

            dTrade = df_prices["Trade"].diff().dropna()
            dHedge = df_prices["Hedge"].diff().dropna()

            pnl_df = pd.concat(
                [dTrade.rename("dTrade"), dHedge.rename("dHedge")],
                axis=1
            ).dropna()

            sign_T = 1 if trade_direction == "Long" else -1

            pnl_df["Trade PnL"] = sign_T * trade_units * pnl_df["dTrade"]
            pnl_df["Hedge PnL"] = -hedge_ratio_k * trade_units * pnl_df["dHedge"]
            pnl_df["Portfolio PnL"] = pnl_df["Trade PnL"] + pnl_df["Hedge PnL"]

            return pnl_df

        # --- UI for backtest ---
        if not Sigma_Raw_df.empty and Sigma_Raw_df.shape[1] > 1:

            backtest_labels = Sigma_Raw_df.columns.tolist()

            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                backtest_trade = st.selectbox(
                    "Backtest Trade Instrument",
                    options=backtest_labels,
                    key="backtest_trade"
                )
            with col_bt2:
                backtest_hedge = st.selectbox(
                    "Backtest Hedge Instrument",
                    options=[x for x in backtest_labels if x != backtest_trade],
                    key="backtest_hedge"
                )

            col_bt3, col_bt4, col_bt5 = st.columns(3)
            with col_bt3:
                backtest_trade_dir = st.selectbox(
                    "Trade Direction (for backtest)",
                    ["Long", "Short"],
                    key="backtest_trade_dir"
                )
            with col_bt4:
                backtest_trade_units = st.number_input(
                    "Trade Size (units)",
                    min_value=0.1,
                    value=1.0,
                    step=0.5,
                    key="backtest_trade_units"
                )
            with col_bt5:
                # Default k* from covariance for convenience
                Var_H = Sigma_Raw_df.loc[backtest_hedge, backtest_hedge]
                Cov_TH = Sigma_Raw_df.loc[backtest_trade, backtest_hedge]
                default_k = float(Cov_TH / Var_H) if Var_H > 1e-9 else 0.0

                backtest_k = st.number_input(
                    "Hedge Ratio k (portfolio = T - kH)",
                    value=default_k,
                    step=0.1,
                    format="%.4f",
                    key="backtest_k"
                )

            if st.button("Run Historical Backtest", key="run_backtest"):
                pnl_df = _compute_hedged_pnl_series(
                    trade_label=backtest_trade,
                    hedge_label=backtest_hedge,
                    trade_direction=backtest_trade_dir,
                    trade_units=backtest_trade_units,
                    hedge_ratio_k=backtest_k
                )

                if pnl_df is not None and not pnl_df.empty:

                    trade_vol = pnl_df["Trade PnL"].std() * 100
                    port_vol = pnl_df["Portfolio PnL"].std() * 100
                    vol_red_pct = (1 - port_vol / trade_vol) * 100 if trade_vol > 0 else float("nan")

                    st.markdown("### Volatility Before vs After Hedging")
                    st.markdown(f"""
                    - **Trade-only Volatility:** `{trade_vol:.4f}` Rate %  
                    - **Hedged Portfolio Vol:** `{port_vol:.4f}` Rate %  
                    - **Volatility Reduction:** `{vol_red_pct:.2f}%`
                    """)

                    cumulative = pnl_df.cumsum()

                    fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                    ax_bt.plot(cumulative.index, cumulative["Trade PnL"], label="Trade P&L")
                    ax_bt.plot(cumulative.index, cumulative["Hedge PnL"], label="Hedge P&L")
                    ax_bt.plot(cumulative.index, cumulative["Portfolio PnL"], label="Portfolio P&L", linewidth=2)

                    ax_bt.axhline(0, color="black", linewidth=0.8, linestyle="--")
                    ax_bt.set_title(
                        f"Cumulative P&L Backtest — {backtest_trade_dir} {backtest_trade_units} {backtest_trade} "
                        f"vs Hedge (k={backtest_k:.4f} × {backtest_hedge})"
                    )
                    ax_bt.set_ylabel("Cumulative P&L")
                    ax_bt.grid(True, linestyle=":", alpha=0.5)
                    ax_bt.legend()

                    st.pyplot(fig_bt)
        else:
            st.info("Backtest unavailable: covariance matrix for derivatives is empty.")

# ------------------- Section 9: PCA Factor Shocks & Whole-Instrument Anchoring -------------------
st.header("9. PCA Factor Shocks & Whole-Instrument Anchoring")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =============================================================================
# Helper functions
# =============================================================================

def parse_derivative(label):
    """
    Examples:
    '3M Spread: H26-M26'        -> [('H26', 1), ('M26', -1)]
    '3M Fly: H26-2xM26+U26'     -> [('H26', 1), ('M26', -2), ('U26', 1)]
    '3M Double Fly: H26-3xM26+3xU26-Z26'
                                 -> [('H26',1),('M26',-3),('U26',3),('Z26',-1)]
    """
    expr = label.split(":")[-1].replace(" ", "")
    tokens = re.findall(r'([+-]?)(\d*)x?([A-Z]\d{2})', expr)

    legs = []
    for sign, mult, c in tokens:
        s = -1 if sign == '-' else 1
        m = int(mult) if mult else 1
        legs.append((c, s * m))
    return legs


def eval_derivative(label, curve):
    return sum(w * curve[c] for c, w in parse_derivative(label))


# =============================================================================
# 1. BUILD DERIVATIVE UNIVERSE
# =============================================================================

all_deriv_list = []

if 'spreads_3M_df_raw' in locals():
    all_deriv_list.append(spreads_3M_df_raw.rename(columns=lambda x: f"3M Spread: {x}"))

if 'butterflies_3M_df' in locals():
    all_deriv_list.append(butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"))

if 'double_butterflies_3M_df' in locals():
    all_deriv_list.append(double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"))

if 'spreads_6M_df' in locals():
    all_deriv_list.append(spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"))

if 'butterflies_6M_df' in locals():
    all_deriv_list.append(butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"))

if not all_deriv_list:
    st.error("Derivative data not found. Run earlier sections first.")
    st.stop()

all_deriv_df = pd.concat(all_deriv_list, axis=1)

Sigma_raw, deriv_aligned, loadings_gen = calculate_derivatives_covariance_generalized(
    all_deriv_df, scores, eigenvalues, pc_count
)

# =============================================================================
# 2. SELECT ANCHOR
# =============================================================================

anchor_label = st.selectbox(
    "Select Instrument to Anchor:",
    sorted(all_deriv_df.columns),
    key="section9_anchor_final"
)

# =============================================================================
# 3. RUN ANCHORED PCA SHOCK
# =============================================================================

if st.button("Run Whole-Instrument Anchor Shock"):
    try:
        # ---------------------------------------------------------------------
        # STEP A: PCA FACTOR SHIFT (minimum-norm)
        # ---------------------------------------------------------------------
        mkt_val = all_deriv_df.loc[analysis_dt, anchor_label]

        L = loadings_gen.loc[anchor_label].iloc[:pc_count].values
        sigma = all_deriv_df[anchor_label].std()
        mean = all_deriv_df[anchor_label].mean()

        pca_fair = (scores.loc[analysis_dt].iloc[:pc_count].values @ L) * sigma + mean
        Z_target = (mkt_val - pca_fair) / sigma

        Lm = L.reshape(1, -1)
        delta_PC = (Lm.T @ np.linalg.inv(Lm @ Lm.T) * Z_target).flatten()

        # ---------------------------------------------------------------------
        # STEP B: PROPAGATE TO 3M SPREAD DNA
        # ---------------------------------------------------------------------
        std3 = spreads_3M_df_clean.std()
        L_sp = loadings_spread.values[:, :pc_count]

        delta_spreads = (L_sp @ delta_PC) * std3.values

        base_spreads = historical_spreads_3M_df.loc[analysis_dt].filter(like="(PCA)")
        base_spreads.index = [
            c.replace(" (PCA)", "").replace("3M Spread: ", "") for c in base_spreads.index
        ]

        shocked_spreads = base_spreads + delta_spreads

        # ---------------------------------------------------------------------
        # STEP C: REBUILD OUTRIGHT CURVE (single pivot)
        # ---------------------------------------------------------------------
        outr_mkt = analysis_curve_df.loc[analysis_dt]

        pivot = contract_labels[len(contract_labels) // 2]
        shocked_out = pd.Series(index=contract_labels, dtype=float)
        shocked_out[pivot] = outr_mkt[pivot]

        # forward
        p_idx = contract_labels.index(pivot)
        for i in range(p_idx + 1, len(contract_labels)):
            p, c = contract_labels[i - 1], contract_labels[i]
            shocked_out[c] = shocked_out[p] - shocked_spreads[f"{p}-{c}"]

        # backward
        for i in range(p_idx - 1, -1, -1):
            c, n = contract_labels[i], contract_labels[i + 1]
            shocked_out[c] = shocked_out[n] + shocked_spreads[f"{c}-{n}"]

        # ---------------------------------------------------------------------
        # STEP D: ENFORCE WHOLE-INSTRUMENT CONSTRAINT (CORRECT WAY)
        # ---------------------------------------------------------------------
        legs = parse_derivative(anchor_label)

        inst_val = sum(w * shocked_out[c] for c, w in legs)
        residual = mkt_val - inst_val

        norm = sum(w * w for _, w in legs)
        for c, w in legs:
            shocked_out[c] += (w / norm) * residual

        # ---------------------------------------------------------------------
        # STEP E: PLOT OUTRIGHT CURVE
        # ---------------------------------------------------------------------
        st.success(f"Successfully anchored: {anchor_label}")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(contract_labels, outr_mkt, label="Market", marker='o', alpha=0.35)
        ax.plot(contract_labels, shocked_out, label="Shocked (Anchored)", marker='x', linestyle='--')

        for c, _ in legs:
            ax.scatter(c, outr_mkt[c], s=160, color='red', zorder=5)

        ax.set_title(f"Whole-Instrument Convergence: {anchor_label}")
        ax.legend()
        st.pyplot(fig)

        # ---------------------------------------------------------------------
        # STEP F: FAMILY SNAPSHOTS
        # ---------------------------------------------------------------------
        st.subheader("Family Impact Analysis")

        shocked_derivs = compute_all_derivatives_from_outrights_row(
            contract_labels, shocked_out
        )

        families = [
            ("3M Spread", "historical_spreads_3M_df", "3M_spreads"),
            ("3M Butterfly", "historical_butterflies_3M_df", "3M_flies"),
            ("3M Double Fly", "historical_double_butterflies_3M_df", "3M_dbf"),
            ("6M Spread", "historical_spreads_6M_df", "6M_spreads"),
            ("6M Butterfly", "historical_butterflies_6M_df", "6M_flies"),
        ]

        for label, hist_name, key in families:
            hist_df = locals().get(hist_name)
            if hist_df is not None and not hist_df.empty:
                with st.expander(f"View {label} Impact"):
                    plot_shock_derivative_snapshot(
                        hist_df,
                        label,
                        shocked_derivs[key],
                        analysis_dt,
                        pc_count,
                        f"(Anchored to {anchor_label})"
                    )

    except Exception as e:
        st.error(f"Convergence failed: {e}")


# ------------------- Section 10: Precision Adaptive Envelopes & Stats -------------------
st.header("10. Precision Adaptive Envelopes (1σ & 2σ)")

# 1. Determine total historical length for the default slider value
sample_df = locals().get("historical_spreads_3M_df")
total_hist_days = len(sample_df) if sample_df is not None else 252

# --- Explanation of Resid and Z-Score ---
st.info("""
**Definitions for Precision Trading:**
* **Resid (bps):** The raw gap between Market and Model. (+ is Rich, - is Cheap).
* **Max Sigma:** The peak volatility 'benchmark' for that specific instrument.
* **Z-Score:** The 'Severity' of the mispricing. Z > 2.0 means the Resid is more than twice the peak historical noise.
""")

# 2. Slider: Defaulting to the MAX historical range
lookback_selection = st.slider(
    "Volatility Lookback Window (Days):", 
    min_value=10, 
    max_value=total_hist_days, 
    value=total_hist_days, 
    key="p10_precision_vfinal",
    help="Default is the full history. This finds the 'peak' noise level reached by each instrument."
)

def plot_with_stats_table(df, label, analysis_dt, window):
    """
    Computes rolling max volatility, plots 1/2 sigma bands, 
    and generates a detailed statistical table.
    """
    # Identify instruments
    instruments = [c.replace(" (Original)", "") for c in df.columns if " (Original)" in c]
    if not instruments: return
    
    data_list = []
    for inst in instruments:
        orig_col, pca_col = f"{inst} (Original)", f"{inst} (PCA)"
        
        if orig_col in df.columns and pca_col in df.columns:
            # 1. Full Residual Series (bps)
            full_res_bps = (df[orig_col] - df[pca_col]) * 100 
            
            # 2. Calculate Rolling Sigma based on slider
            rolling_sigma = full_res_bps.rolling(window=window, min_periods=min(10, window)).std()
            
            # 3. Find the MAX sigma from that rolling series
            max_sigma_bps = rolling_sigma.max()
            
            # Fallback
            if np.isnan(max_sigma_bps): max_sigma_bps = full_res_bps.std()
            
            curr_mkt = df.loc[analysis_dt, orig_col]
            curr_pca = df.loc[analysis_dt, pca_col]
            curr_res = (curr_mkt - curr_pca) * 100
            
            z_score = curr_res / max_sigma_bps if max_sigma_bps > 0 else 0
            
            # Signal Logic
            if z_score > 2: signal = "EXTREME RICH"
            elif z_score > 1: signal = "RICH"
            elif z_score < -2: signal = "EXTREME CHEAP"
            elif z_score < -1: signal = "CHEAP"
            else: signal = "FAIR"

            data_list.append({
                "Instrument": inst,
                "Market": curr_mkt, "Fair": curr_pca, "Resid (bps)": curr_res,
                "Max Sigma (bps)": max_sigma_bps,
                "Z-Score": z_score,
                "Signal": signal,
                "U2": curr_pca + (2 * max_sigma_bps / 100),
                "U1": curr_pca + (1 * max_sigma_bps / 100),
                "L1": curr_pca - (1 * max_sigma_bps / 100),
                "L2": curr_pca - (2 * max_sigma_bps / 100),
            })
    
    if not data_list: return
    plot_df = pd.DataFrame(data_list)
    x = range(len(plot_df))
    
    # --- CHART ---
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(x, plot_df["U2"], color='#c0392b', linestyle=':', alpha=0.7, label="Max 2σ (Outlier)")
    ax.plot(x, plot_df["L2"], color='#c0392b', linestyle=':', alpha=0.7)
    ax.plot(x, plot_df["U1"], color='#e67e22', linestyle='--', alpha=0.5, label="Max 1σ (Normal)")
    ax.plot(x, plot_df["L1"], color='#e67e22', linestyle='--', alpha=0.5)
    ax.plot(x, plot_df["Fair"], color='black', label="PCA Fair", linewidth=1.2, alpha=0.4)
    ax.plot(x, plot_df["Market"], color='#2980b9', marker='o', linewidth=2.5, label="Market", markersize=8)
    
    # BPS Annotations
    for i, row in plot_df.iterrows():
        text_color = 'red' if abs(row['Z-Score']) > 1 else 'blue'
        ax.annotate(f"{row['Resid (bps)']:.1f}", (i, row['Market']), 
                    xytext=(0, 12), textcoords="offset points", 
                    ha='center', fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(x); ax.set_xticklabels(plot_df["Instrument"], rotation=45, ha='right')
    ax.set_title(f"{label} Curve: Statistical Boundaries (Max-Rolling {window}d)", fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(True, alpha=0.15)
    st.pyplot(fig)

    # --- TABLE ---
    st.write(f"**{label} Precision Statistics**")
    
    plot_df["Dist to 2σ (bps)"] = np.where(
        plot_df["Resid (bps)"] > 0, 
        plot_df["Resid (bps)"] - (2 * plot_df["Max Sigma (bps)"]),
        plot_df["Resid (bps)"] + (2 * plot_df["Max Sigma (bps)"])
    )

    st.dataframe(
        plot_df[["Instrument", "Resid (bps)", "Max Sigma (bps)", "Z-Score", "Signal", "Dist to 2σ (bps)"]]
        .style.background_gradient(subset=['Z-Score'], cmap='RdYlGn_r', vmin=-3, vmax=3)
        .applymap(lambda x: 'color: red; font-weight: bold' if "EXTREME" in str(x) else '', subset=['Signal'])
        .format({"Resid (bps)": "{:.2f}", "Max Sigma (bps)": "{:.2f}", "Z-Score": "{:.2f}", "Dist to 2σ (bps)": "{:.2f}"}),
        use_container_width=True
    )
    st.divider()

# --- EXECUTE FOR ALL FAMILIES ---
families = [
    ("3M Spread", "historical_spreads_3M_df"), 
    ("6M Spread", "historical_spreads_6M_df"),
    ("3M Butterfly", "historical_butterflies_3M_df"), 
    ("6M Butterfly", "historical_butterflies_6M_df"),
    ("3M Double Fly", "historical_double_butterflies_3M_df"),
    ("6M Double Fly", "historical_double_butterflies_6M_df")
]

for label, var in families:
    df_found = locals().get(var)
    if df_found is not None and not df_found.empty:
        plot_with_stats_table(df_found, label, analysis_dt, lookback_selection)
# ============================
# SECTION 11 — KALMAN FILTERED PCA FAIR CURVE
# ============================

st.header("11. Kalman-Filtered PCA Fair Curve")

st.markdown("""
This section builds a **dynamic fair value curve** by applying a **Kalman filter**
to PCA factor scores.  
The **output snapshot is identical to Section 5**, but uses **noise-filtered factors**.
""")

# ----------------------------
# Helper functions (LOCAL)
# ----------------------------

def _estimate_phi_ar1(series, clip=(0.70, 0.995)):
    x = np.asarray(series)
    if len(x) < 10:
        return 0.95
    x_lag = x[:-1]
    x_now = x[1:]
    denom = np.dot(x_lag, x_lag)
    if denom < 1e-8:
        return 0.95
    phi = np.dot(x_now, x_lag) / denom
    return float(np.clip(phi, clip[0], clip[1]))


def _kalman_filter_1d(observed, phi, q, r):
    n = len(observed)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = observed[0]
    P[0] = 1.0

    for t in range(1, n):
        x_pred = phi * x_hat[t - 1]
        P_pred = phi**2 * P[t - 1] + q
        K = P_pred / (P_pred + r)
        x_hat[t] = x_pred + K * (observed[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x_hat


# ----------------------------
# UI toggle
# ----------------------------

use_kalman = st.checkbox(
    "Use Kalman-Filtered PCA Factors",
    value=True,
    key="use_kalman_section11"
)

# ----------------------------
# Apply Kalman to PCA scores
# ----------------------------

kalman_scores = scores.copy()
phi_rows = []

if use_kalman:
    for i in range(pc_count):
        pc_name = kalman_scores.columns[i]
        raw_series = kalman_scores[pc_name].values

        phi = _estimate_phi_ar1(raw_series)
        var = np.var(raw_series)

        q = 0.01 * var   # process noise
        r = 1.00 * var   # measurement noise

        kalman_scores[pc_name] = _kalman_filter_1d(
            raw_series, phi=phi, q=q, r=r
        )

        phi_rows.append({
            "PC": pc_name,
            "Estimated φ": round(phi, 4)
        })

    st.subheader("Estimated PCA Factor Persistence")
    st.dataframe(pd.DataFrame(phi_rows), use_container_width=True)

else:
    st.info("Raw PCA scores are used (Kalman disabled).")

# ----------------------------
# Reconstruct 3M spreads
# ----------------------------

data_mean = spreads_3M_df_clean.mean()
data_std = spreads_3M_df_clean.std()

scores_used = (
    kalman_scores.values[:, :pc_count]
    if use_kalman
    else scores.values[:, :pc_count]
)

loadings_used = loadings_spread.values[:, :pc_count]

reconstructed_scaled = scores_used @ loadings_used.T

reconstructed_spreads_3M_kf = pd.DataFrame(
    reconstructed_scaled * data_std.values + data_mean.values,
    index=spreads_3M_df_clean.index,
    columns=spreads_3M_df_clean.columns
)

# ----------------------------
# Reconstruct outrights & derivatives
# ----------------------------

(
    historical_outrights_kf,
    historical_spreads_3M_kf,
    historical_butterflies_3M_kf,
    historical_spreads_6M_kf,
    historical_butterflies_6M_kf,
    historical_spreads_12M_kf,
    historical_butterflies_12M_kf,
    historical_double_butterflies_3M_kf,
    historical_double_butterflies_6M_kf,
    historical_double_butterflies_12M_kf,
    _
) = reconstruct_prices_and_derivatives(
    analysis_curve_df,
    reconstructed_spreads_3M_kf,
    spreads_3M_df_raw,
    spreads_6M_df,
    butterflies_3M_df,
    butterflies_6M_df,
    spreads_12M_df,
    butterflies_12M_df,
    double_butterflies_3M_df,
    double_butterflies_6M_df,
    double_butterflies_12M_df
)

# ----------------------------
# Snapshot output (Section-5 style)
# ----------------------------

st.subheader("11.1 Kalman Fair Curve Snapshot — 3M Spreads")
plot_snapshot(
    historical_spreads_3M_kf,
    derivative_type="3M Spread",
    current_date=analysis_dt,
    pc_count=pc_count
)

st.subheader("11.2 Kalman Fair Curve Snapshot — 3M Flies")
plot_snapshot(
    historical_butterflies_3M_kf,
    derivative_type="3M Fly",
    current_date=analysis_dt,
    pc_count=pc_count
)

st.subheader("11.3 Kalman Fair Curve Snapshot — 3M Double Flies")
plot_snapshot(
    historical_double_butterflies_3M_kf,
    derivative_type="3M Double Fly",
    current_date=analysis_dt,
    pc_count=pc_count
)

# ============================
# END SECTION 11
# ============================================================
# SECTION 12: TRADE STRUCTURING & PCA MISPRICING CAPTURE
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# 12.0 EXPRESSION QUALITY OF THE SELECTED INSTRUMENT
# -------------------------------------------------------------------

def compute_expression_quality(instrument, factor_sensitivities_df, Sigma_Raw_df, mispricing_series):
    """
    Absolute quality of ONE instrument as a trading vehicle
    """

    betas = factor_sensitivities_df.loc[instrument]
    mispricing = abs(mispricing_series.get(instrument, np.nan))

    # Factor purity: single-factor vs mixed exposure
    factor_purity = betas.abs().max() / betas.abs().sum()

    # Avg absolute correlation vs entire universe
    avg_abs_corr = Sigma_Raw_df.corr().abs().mean().get(instrument, np.nan)

    expression_quality = mispricing * factor_purity / (1 + avg_abs_corr)

    return {
        "Mispricing (Rate %)": mispricing,
        "Dominant Factor": betas.abs().idxmax(),
        "Factor Purity": factor_purity,
        "Avg Abs Correlation": avg_abs_corr,
        "Expression Quality Score": expression_quality
    }


# -------------------------------------------------------------------
# 12.1 ALTERNATIVE EXPRESSIONS OF THE SAME DISTORTION
# -------------------------------------------------------------------

def find_alternative_expressions(
    selected_instrument,
    instrument_universe_df,
    factor_sensitivities_df,
    Sigma_Raw_df,
    mispricing_series,
    top_n=5
):
    T = selected_instrument
    T_betas = factor_sensitivities_df.loc[T]
    T_mis = abs(mispricing_series.get(T, np.nan))

    maturity_tag = (
        "3M" if "3M" in T else
        "6M" if "6M" in T else
        "12M" if "12M" in T else ""
    )

    local_universe = instrument_universe_df[
        instrument_universe_df["Instrument"].str.contains(maturity_tag)
    ]

    rows = []

    for C in local_universe["Instrument"]:
        if C == T or C not in factor_sensitivities_df.index:
            continue

        C_betas = factor_sensitivities_df.loc[C]

        # Factor alignment (cosine similarity)
        alignment = np.dot(T_betas, C_betas) / (
            np.linalg.norm(T_betas) * np.linalg.norm(C_betas)
        )

        # Pairwise correlation vs selected instrument
        corr_vs_selected = Sigma_Raw_df.loc[T, C] / np.sqrt(
            Sigma_Raw_df.loc[T, T] * Sigma_Raw_df.loc[C, C]
        )

        relative_score = T_mis * abs(alignment) / (1 + abs(corr_vs_selected))

        rows.append({
            "Alternative Instrument": C,
            "Factor Alignment": alignment,
            "Correlation vs Selected": corr_vs_selected,
            "Relative Expression Score": relative_score
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Relative Expression Score", ascending=False).head(top_n)


# -------------------------------------------------------------------
# 12.2 FACTOR-ISOLATED COMBO TRADE
# -------------------------------------------------------------------

def build_factor_isolated_combo(
    primary_instr,
    hedge_instr,
    factor_sensitivities_df,
    Sigma_Raw_df,
    mispricing_series
):
    T_betas = factor_sensitivities_df.loc[primary_instr]
    H_betas = factor_sensitivities_df.loc[hedge_instr]

    dominant_factor = T_betas.abs().idxmax()

    # Hedge ratio removes dominant factor
    k = T_betas[dominant_factor] / H_betas[dominant_factor]

    residuals = T_betas - k * H_betas

    var_T = Sigma_Raw_df.loc[primary_instr, primary_instr]
    var_H = Sigma_Raw_df.loc[hedge_instr, hedge_instr]
    cov_TH = Sigma_Raw_df.loc[primary_instr, hedge_instr]

    residual_var = var_T + k**2 * var_H - 2 * k * cov_TH
    residual_vol = np.sqrt(max(residual_var, 0)) * 100

    direction = (
        "Sell / Receive" if mispricing_series.get(primary_instr, 0) > 0
        else "Buy / Pay"
    )

    return {
        "Primary Instrument": primary_instr,
        "Hedge Instrument": hedge_instr,
        "Trade Direction": direction,
        "Target Factor": dominant_factor,
        "Hedge Ratio (k)": k,
        "Residual Level": residuals.get("Level (Whole Curve Shift)", np.nan),
        "Residual Slope": residuals.get("Slope (Steepening/Flattening)", np.nan),
        "Residual Curvature": residuals.get("Curvature (Fly Risk)", np.nan),
        "Residual Risk (Rate %)": residual_vol
    }


# -------------------------------------------------------------------
# 12.3 PCA MISPRICING CAPTURE (NOT $ PnL)
# -------------------------------------------------------------------

def backtest_pca_mispricing_capture(
    primary_instr,
    hedge_instr,
    k,
    historical_derivatives_list,
    holding_days=5
):
    mis_ts = {}

    for df in historical_derivatives_list:
        for col in df.columns:
            if col.endswith("(Original)"):
                base = col.replace(" (Original)", "")
                pca_col = col.replace("(Original)", "(PCA)")
                if pca_col in df.columns:
                    mis_ts[base] = (df[col] - df[pca_col]) * 100

    mis_df = pd.DataFrame(mis_ts).dropna()

    if primary_instr not in mis_df or hedge_instr not in mis_df:
        return None

    combo_mis = mis_df[primary_instr] - k * mis_df[hedge_instr]
    capture = combo_mis - combo_mis.shift(-holding_days)
    capture = capture.dropna()
    cum_capture = capture.cumsum()

    return {
        "Total Mispricing Captured (Rate %)": cum_capture.iloc[-1],
        "Mean-Reversion Sharpe": capture.mean() / capture.std() * np.sqrt(252),
        "Hit Rate": (capture > 0).mean(),
        "Max Drawdown (Rate %)": (cum_capture - cum_capture.cummax()).min()
    }


# -------------------------------------------------------------------
# 12.4 STREAMLIT UI + EXPLANATIONS
# -------------------------------------------------------------------

st.header("12. Trade Structuring & PCA Mispricing Capture")

with st.expander("ℹ️ How to read Section 12 (definitions & formulas)", expanded=False):
    st.markdown(r"""
### Mispricing (Rate %)
\[
(\text{Market} - \text{PCA Fair}) \times 100
\]

### Factor Purity
\[
\frac{\max(|\beta_L|,|\beta_S|,|\beta_C|)}
{|\beta_L|+|\beta_S|+|\beta_C|}
\]

### Avg Abs Correlation
\[
\frac{1}{N}\sum_{j\neq i} |\rho(i,j)|
\]
High = proxy / crowded (BAD)

### Expression Quality Score
\[
\frac{|\text{Mispricing}|\times \text{Factor Purity}}
{1+\text{Avg Abs Corr}}
\]

### Factor Alignment
Cosine similarity of factor vectors (≈1 means same idea)

### Correlation vs Selected
\[
\rho(i,j)
\]
High = GOOD (same regional distortion)

### PCA Mispricing Capture (NOT $ PnL)
\[
(\text{Mis}_T - k\text{Mis}_H)_t -
(\text{Mis}_T - k\text{Mis}_H)_{t+N}
\]
Units are **Rate %**, not dollars.
""")

selected_instr = st.selectbox(
    "1️⃣ Select instrument where you see distortion",
    instrument_universe_df["Instrument"].values
)

quality = compute_expression_quality(
    selected_instr, factor_sensitivities_df, Sigma_Raw_df, mispricing_series
)

st.subheader("A. Instrument quality")
st.table(pd.DataFrame(quality, index=["Value"]).T)

alt_df = find_alternative_expressions(
    selected_instr,
    instrument_universe_df,
    factor_sensitivities_df,
    Sigma_Raw_df,
    mispricing_series
)

st.subheader("B. Alternative expressions")
st.dataframe(alt_df, use_container_width=True)

trade_instr = st.selectbox(
    "2️⃣ Choose instrument to trade",
    alt_df["Alternative Instrument"].values
)

combo = build_factor_isolated_combo(
    selected_instr,
    trade_instr,
    factor_sensitivities_df,
    Sigma_Raw_df,
    mispricing_series
)

st.subheader("C. Structured trade")
st.table(pd.DataFrame(combo, index=["Value"]).T)

holding_days = st.slider("Holding period (days)", 1, 20, 5)

stats = backtest_pca_mispricing_capture(
    selected_instr,
    trade_instr,
    combo["Hedge Ratio (k)"],
    all_historical_derivatives_list,
    holding_days
)

if stats:
    st.subheader("D. PCA mispricing capture (NOT $ PnL)")
    st.table(pd.DataFrame(stats, index=["Value"]).T)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Instrument Level Curves (Separate Views, Actual Levels)
# ---------------------------------------------------------------------

st.subheader("Instrument Level Curves (Separate Views, Actual Levels)")

# --- Build historical ORIGINAL level universe ---
historical_levels_df = pd.concat(all_historical_derivatives_list, axis=1)

primary_col = f"{selected_instr} (Original)"
hedge_col   = f"{trade_instr} (Original)"
k_star      = combo["Hedge Ratio (k)"]

if primary_col not in historical_levels_df.columns or hedge_col not in historical_levels_df.columns:
    st.warning("Original level series not available for selected instruments.")
else:
    # --- Extract series ---
    primary_series = historical_levels_df[primary_col].dropna()
    hedge_series   = historical_levels_df[hedge_col].dropna()

    # --- Align dates ---
    common_idx = primary_series.index.intersection(hedge_series.index)

    if len(common_idx) < 10:
        st.warning("Not enough overlapping history for level curves.")
    else:
        primary_series = primary_series.loc[common_idx]
        hedge_series   = hedge_series.loc[common_idx]

        # --- Hedged synthetic instrument (LEVEL) ---
        hedged_series = primary_series - k_star * hedge_series

        # =========================
        # PRIMARY INSTRUMENT
        # =========================
        fig1, ax1 = plt.subplots(figsize=(15, 4))
        ax1.plot(primary_series.index, primary_series.values, linewidth=2.5)
        ax1.set_title(f"Primary Instrument Level: {selected_instr}", fontsize=14)
        ax1.set_ylabel("Instrument Level")
        ax1.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig1)

        # =========================
        # HEDGE INSTRUMENT
        # =========================
        fig2, ax2 = plt.subplots(figsize=(15, 4))
        ax2.plot(
            hedge_series.index,
            hedge_series.values,
            linewidth=2.5,
            linestyle="--"
        )
        ax2.set_title(f"Hedge Instrument Level: {trade_instr}", fontsize=14)
        ax2.set_ylabel("Instrument Level")
        ax2.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig2)

        # =========================
        # HEDGED SYNTHETIC
        # =========================
        fig3, ax3 = plt.subplots(figsize=(15, 4))
        ax3.plot(
            hedged_series.index,
            hedged_series.values,
            linewidth=2.8
        )
        ax3.set_title(
            f"Hedged Synthetic Instrument Level (Primary − {k_star:.3f} × Hedge)",
            fontsize=14
        )
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Instrument Level")
        ax3.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig3)



# ======================
# END SECTION 12
# ==========================================================
# 5.d — FILTERED MISPRICING TABLE + FAMILY FILTER + HEDGES
# ==========================================================

st.subheader("5.d Mispricing Filter + Family Selection + Hedge Suggestions")

# Requires:
# mispricing_series  -> from calculate_derivative_mispricings()
# Sigma_Raw_df       -> PCA reconstructed covariance from Section 7

if mispricing_series is not None and len(mispricing_series) > 0:

    # ------------------------------------------------------
    # BUILD MISPRICING DATAFRAME
    # ------------------------------------------------------
    mispricing_df = mispricing_series.reset_index()
    mispricing_df.columns = ["Instrument", "Mispricing (Rate %)"]

    # --- classify derivative family ---
    def classify_family(name):

        if "3M" in name:
            tenor = "3M"
        elif "6M" in name:
            tenor = "6M"
        elif "12M" in name:
            tenor = "12M"
        else:
            tenor = "Other"

        if "Double Fly" in name:
            typ = "Double Fly"
        elif "Fly" in name:
            typ = "Fly"
        elif "Spread" in name:
            typ = "Spread"
        else:
            typ = "Other"

        return f"{tenor} {typ}".strip()

    mispricing_df["Family"] = mispricing_df["Instrument"].apply(classify_family)
    mispricing_df["Abs Mispricing"] = mispricing_df["Mispricing (Rate %)"].abs()

    # ------------------------------------------------------
    # FILTER CONTROLS
    # ------------------------------------------------------
    col1, col2 = st.columns(2)

    # --- Threshold slider ---
    with col1:
        max_range = float(np.nanmax(np.abs(mispricing_series.values))) if len(mispricing_series) > 0 else 5.0

        threshold_rate = st.slider(
            "Minimum Absolute Mispricing Threshold (Rate %)",
            min_value=0.0,
            max_value=max_range if max_range > 0 else 5.0,
            value=min(0.10, max_range) if max_range > 0 else 0.10,
            step=0.01
        )

    # --- Family filter selector ---
    with col2:
        available_families = sorted(mispricing_df["Family"].unique().tolist())

        selected_families = st.multiselect(
            "Select Derivative Families",
            options=available_families,
            default=available_families,
            help="Filter by tenor and derivative type"
        )

    # ------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------
    filtered_df = mispricing_df[
        (mispricing_df["Abs Mispricing"] >= threshold_rate) &
        (mispricing_df["Family"].isin(selected_families))
    ].sort_values("Abs Mispricing", ascending=False)

    # ------------------------------------------------------
    # MINIMUM VARIANCE HEDGE ENGINE (PCA COVARIANCE BASED)
    # ------------------------------------------------------
    def find_best_hedge(trade_label, Sigma):
        """
        Minimum Variance Hedge:
            k* = Cov(T,H) / Var(H)
            Residual Var = Var(T) - k*Cov(T,H)
        """
        if Sigma is None or Sigma.empty:
            return None, None, None, None

        if trade_label not in Sigma.index:
            return None, None, None, None

        Var_T = Sigma.loc[trade_label, trade_label]
        best_residual = np.inf
        best_hedge = None
        best_k = None

        for hedge in Sigma.columns:
            if hedge == trade_label:
                continue

            Var_H = Sigma.loc[hedge, hedge]
            Cov_TH = Sigma.loc[trade_label, hedge]

            if Var_H <= 1e-9:
                continue

            k = Cov_TH / Var_H
            residual_var = Var_T - k * Cov_TH
            residual_var = max(residual_var, 0)

            if residual_var < best_residual:
                best_residual = residual_var
                best_hedge = hedge
                best_k = k

        if best_hedge is None:
            return None, None, None, None

        residual_vol = np.sqrt(best_residual) * 100
        action = "Short Hedge" if best_k > 0 else "Long Hedge"

        return best_hedge, abs(best_k), residual_vol, action

    # ------------------------------------------------------
    # COMPUTE HEDGE SUGGESTIONS
    # ------------------------------------------------------
    hedge_list = []
    hedge_ratio_list = []
    residual_list = []
    action_list = []

    for instr in filtered_df["Instrument"]:

        if 'Sigma_Raw_df' in globals() and not Sigma_Raw_df.empty:
            hedge, k, resid, action = find_best_hedge(instr, Sigma_Raw_df)
        else:
            hedge, k, resid, action = None, None, None, None

        hedge_list.append(hedge if hedge else "N/A")
        hedge_ratio_list.append(k if k else np.nan)
        residual_list.append(resid if resid else np.nan)
        action_list.append(action if action else "N/A")

    filtered_df["Suggested Hedge"] = hedge_list
    filtered_df["Hedge Ratio |k*|"] = hedge_ratio_list
    filtered_df["Hedge Action"] = action_list
    filtered_df["Residual Risk After Hedge (Rate %)"] = residual_list

    # ------------------------------------------------------
    # DISPLAY OUTPUT
    # ------------------------------------------------------
    if filtered_df.empty:
        st.info("No instruments match selected filters.")
    else:
        st.metric("Filtered Instruments", len(filtered_df))

        st.caption("""
Hedge Basis: **Minimum Variance Hedge using PCA Risk Model**

• PCA reconstructed covariance matrix  
• Hedge ratio: k* = Cov(trade, hedge) / Var(hedge)  
• Hedge selected to minimize residual volatility  
• Residual risk shows remaining exposure after hedge
""")

        st.dataframe(
            filtered_df.drop(columns=["Abs Mispricing"]).style.format({
                "Mispricing (Rate %)": "{:.4f}",
                "Hedge Ratio |k*|": "{:.4f}",
                "Residual Risk After Hedge (Rate %)": "{:.4f}"
            }),
            use_container_width=True
        )

else:
    st.info("Mispricing data not available.")
    # ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
# 5.e — TRADE RELATIONSHIP EXPLORER
# (Rolling Lookback + Correlation + Lead/Lag + Granger)
# ==========================================================

st.subheader("5.e Trade Relationship Explorer (Correlation + Lead/Lag + Granger Causality)")

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

try:

    # ------------------------------------------------------
    # BUILD MASTER DERIVATIVE TIMESERIES MATRIX
    # ------------------------------------------------------
    def extract_original_columns(df):
        if df is None or df.empty:
            return pd.DataFrame()

        cols = [c for c in df.columns if "(Original)" in c]
        if not cols:
            return pd.DataFrame()

        clean = df[cols].copy()
        clean.columns = [c.replace(" (Original)", "") for c in cols]
        return clean

    all_derivatives_list = [
        extract_original_columns(historical_spreads_3M_df),
        extract_original_columns(historical_butterflies_3M_df),
        extract_original_columns(historical_double_butterflies_3M_df),
        extract_original_columns(historical_spreads_6M_df),
        extract_original_columns(historical_butterflies_6M_df),
        extract_original_columns(historical_double_butterflies_6M_df),
        extract_original_columns(historical_spreads_12M_df),
        extract_original_columns(historical_butterflies_12M_df),
        extract_original_columns(historical_double_butterflies_12M_df),
    ]

    derivatives_ts = pd.concat(all_derivatives_list, axis=1).dropna(axis=1, how="all")

    if derivatives_ts.empty:
        st.info("No derivative time series available.")
        st.stop()

    # ------------------------------------------------------
    # ROLLING LOOKBACK WINDOW (FIXED ROLLING BACK)
    # ------------------------------------------------------
    total_days_available = len(derivatives_ts)

    if total_days_available < 30:
        st.warning("Not enough data for rolling analysis (minimum 30 days required).")
        st.stop()

    lookback_days = st.slider(
        "Rolling Lookback Window (Days Used for Analysis)",
        min_value=30,
        max_value=total_days_available,
        value=min(250, total_days_available)
    )

    # use most recent N days
    derivatives_ts = derivatives_ts.tail(lookback_days)

    # ------------------------------------------------------
    # TRADE SELECTION
    # ------------------------------------------------------
    trade_selected = st.selectbox(
        "Select Trade",
        sorted(derivatives_ts.columns.tolist())
    )

    # ------------------------------------------------------
    # FILTER CONTROLS
    # ------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        corr_threshold = st.slider(
            "Min |Correlation|",
            0.0, 1.0, 0.50, 0.05
        )

    with col2:
        max_lag_days = st.slider(
            "Max Lead/Lag Days",
            1, 20, 5
        )

    with col3:
        granger_p_threshold = st.slider(
            "Max Granger p-value",
            0.01, 1.0, 0.05, 0.01
        )

    # ------------------------------------------------------
    # FFT LEAD/LAG DETECTION
    # ------------------------------------------------------
    def compute_lead_lag_fft(trade_series, other_series, max_lag):

        df = pd.concat([trade_series, other_series], axis=1).dropna()
        if len(df) < 50:
            return None, 0

        x = (df.iloc[:,0] - df.iloc[:,0].mean()) / df.iloc[:,0].std()
        y = (df.iloc[:,1] - df.iloc[:,1].mean()) / df.iloc[:,1].std()

        corr = np.correlate(x, y, mode="full")
        lags = np.arange(-len(x)+1, len(x))

        mask = (lags >= -max_lag) & (lags <= max_lag)
        corr = corr[mask]
        lags = lags[mask]

        idx = np.argmax(np.abs(corr))
        return corr[idx] / len(x), lags[idx]

    # ------------------------------------------------------
    # GRANGER CAUSALITY TEST
    # ------------------------------------------------------
    def granger_test(trade_series, other_series, max_lag=5):

        df = pd.concat([trade_series, other_series], axis=1).dropna()
        if len(df) < 100:
            return None

        try:
            result = grangercausalitytests(df, maxlag=max_lag, verbose=False)
            pvals = [result[i+1][0]["ssr_ftest"][1] for i in range(max_lag)]
            return min(pvals)
        except:
            return None

    # ------------------------------------------------------
    # COMPUTE RELATIONSHIPS
    # ------------------------------------------------------
    trade_series_full = derivatives_ts[trade_selected].dropna()
    aligned_df = derivatives_ts.loc[trade_series_full.index]

    results = []

    for col in aligned_df.columns:

        if col == trade_selected:
            continue

        other_series = aligned_df[col].dropna()
        common_idx = trade_series_full.index.intersection(other_series.index)

        if len(common_idx) < 100:
            continue

        trade_series = trade_series_full.loc[common_idx]
        other_series = other_series.loc[common_idx]

        # correlation
        corr = trade_series.corr(other_series)

        # lag detection
        best_corr, best_lag = compute_lead_lag_fft(
            trade_series, other_series, max_lag_days
        )

        # granger test
        p_val = granger_test(trade_series, other_series, max_lag_days)

        if best_lag > 0:
            relation = "Trade Leads"
        elif best_lag < 0:
            relation = "Trade Follows"
        else:
            relation = "Simultaneous"

        if p_val is not None:
            if p_val < 0.01:
                predict_strength = "Very Strong"
            elif p_val < 0.05:
                predict_strength = "Predictive"
            else:
                predict_strength = "Weak"
        else:
            predict_strength = "N/A"

        results.append({
            "Trade": trade_selected,
            "Instrument": col,
            "Correlation": corr,
            "Lag (Days)": best_lag,
            "Relationship": relation,
            "Granger p-value": p_val,
            "Predictive Strength": predict_strength,
            "Abs Correlation": abs(corr) if corr is not None else 0
        })

    if not results:
        st.info("No relationships found.")
        st.stop()

    df_results = pd.DataFrame(results)

    # ------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------
    filtered = df_results[
        (df_results["Abs Correlation"] >= corr_threshold) &
        ((df_results["Granger p-value"].isna()) |
         (df_results["Granger p-value"] <= granger_p_threshold))
    ].sort_values("Abs Correlation", ascending=False)

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------
    if filtered.empty:
        st.info("No instruments match filters.")
    else:
        st.metric("Filtered Relationships", len(filtered))

        st.caption("""
• Uses rolling lookback window from selected date range  
• Correlation → co-movement strength  
• Lag → who moves first  
• Granger p-value → predictive power (lower = stronger)
""")

        st.dataframe(
            filtered.drop(columns=["Abs Correlation"]).style.format({
                "Correlation": "{:.3f}",
                "Granger p-value": "{:.4f}"
            }),
            use_container_width=True
        )

except Exception as e:
    st.warning(f"Relationship analysis unavailable: {e}")
#---------------------------------------section 12 end here---------------#
# ============================================================
# SECTION 14 — SR3 FOMC POLICY EXPOSURE ENGINE (CORRECTED)
# ============================================================

st.header("14. Policy Path Interpretation")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# ------------------------------------------------------------
# 1) FOMC calendar (extend as needed)
# ------------------------------------------------------------

FOMC = pd.to_datetime([
"2026-03-18","2026-05-06","2026-06-17","2026-07-29","2026-09-16","2026-11-04","2026-12-16",
"2027-01-27","2027-03-17","2027-04-28","2027-06-16","2027-07-28","2027-09-22","2027-11-03","2027-12-15",
"2028-01-26","2028-03-15","2028-05-03","2028-06-14","2028-07-26","2028-09-20","2028-11-01","2028-12-13",
"2029-01-31","2029-03-20","2029-05-01","2029-06-19","2029-07-31","2029-09-18","2029-11-07","2029-12-18"
])

# ------------------------------------------------------------
# 2) Correct SR3 meeting weights
# A meeting controls the rate AFTER it occurs
# ------------------------------------------------------------

def contract_meeting_weights(start, end):

    total_days = (end - start).days
    weights = {}

    meetings = list(FOMC[(FOMC > start) & (FOMC < end)])
    timeline = [start] + meetings + [end]

    for i in range(len(timeline) - 1):

        s = timeline[i]
        e = timeline[i + 1]
        days = (e - s).days

        # Rate in this interval set by LAST meeting
        if i == 0:
            label = "Before First Meeting"
        else:
            label = meetings[i - 1].strftime("%Y-%m")

        weights[label] = weights.get(label, 0) + days / total_days

    return weights

# Build contract weights
contract_weights = {}

for c, row in expiry_df.iterrows():
    end = row["ExpiryDate"]
    start = end - DateOffset(months=3)
    contract_weights[c] = contract_meeting_weights(start, end)

# ------------------------------------------------------------
# 3) Convert trade label → contract coefficients
# ------------------------------------------------------------

def trade_contract_weights(label):

    key = (label.replace("3M ","")
                .replace("6M ","")
                .replace("12M ","")
                .replace("Spread: ","")
                .replace("Fly: ","")
                .replace("Double Fly: ",""))

    key = key.replace("2x","").replace("3x","")
    parts = [p for p in key.replace("+","-").split("-") if p!=""]

    if len(parts) == 2:
        w = [1,-1]
    elif len(parts) == 3:
        w = [1,-2,1]
    elif len(parts) == 4:
        w = [1,-3,3,-1]
    else:
        return None

    return dict(zip(parts,w))

# ------------------------------------------------------------
# 4) Build trade → meeting exposure matrix
# ------------------------------------------------------------

records = []

for trade in mispricing_series.index:

    cw = trade_contract_weights(trade)
    if cw is None:
        continue

    meeting_exp = {}

    for contract, coef in cw.items():

        if contract not in contract_weights:
            continue

        for m, w in contract_weights[contract].items():
            meeting_exp[m] = meeting_exp.get(m,0) + coef*w

    meeting_exp["Trade"] = trade
    records.append(meeting_exp)

exposure_df = pd.DataFrame(records).set_index("Trade").fillna(0)

st.subheader("Meeting Exposure Matrix (bp change in trade per 1bp meeting repricing)")
st.dataframe(exposure_df)

# ------------------------------------------------------------
# 5) Interactive position builder
# ------------------------------------------------------------

st.subheader("Position Policy Impact")

trades = exposure_df.index.tolist()

c1,c2 = st.columns(2)

with c1:
    tA = st.selectbox("Trade A", trades)
    lA = st.number_input("Lots A", value=100, step=25)

with c2:
    tB = st.selectbox("Trade B", trades, index=min(1,len(trades)-1))
    lB = st.number_input("Lots B", value=0, step=25)

net_exp = exposure_df.loc[tA]*lA + exposure_df.loc[tB]*lB
net_exp = net_exp[net_exp.abs() > 1e-6]

st.subheader("Meeting Sensitivity (bp of trade value per 1bp meeting surprise)")
st.dataframe(net_exp.to_frame("Exposure"))

# ------------------------------------------------------------
# 6) Bar chart
# ------------------------------------------------------------

fig, ax = plt.subplots()
net_exp.plot(kind="bar", ax=ax)
ax.set_ylabel("Trade value change per 1bp repricing")
ax.set_title("Which FOMC meetings drive this position")
st.pyplot(fig)

# ------------------------------------------------------------
# 7) Policy interpretation
# ------------------------------------------------------------

misA = mispricing_series.get(tA,0)*lA
misB = mispricing_series.get(tB,0)*lB
net_mis = misA + misB

st.subheader("Policy Interpretation")

if abs(net_mis) < 1e-6:
    st.info("Position already near fair value")
else:

    dominant = net_exp.abs().idxmax()
    direction = "higher rates than priced" if net_mis < 0 else "lower rates than priced"

    st.markdown(f"""
This position mainly depends on the **{dominant} FOMC meeting**.

To move toward fair value, the market must price **{direction}** around that meeting.

Interpretation:
You are not betting on time passing — you are betting that
the expected policy decision at this specific meeting is mispriced.
""")
