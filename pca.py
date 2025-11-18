import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

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
        
        def plot_snapshot(historical_df, derivative_type, current_date, pc_count):
            """Plots the market vs PCA fair value snapshot for a given derivative type."""
            
            try:
                # 1. Get the snapshot for the selected date
                market_values = historical_df.loc[current_date].filter(like='(Original)')
                pca_fair_values = historical_df.loc[current_date].filter(like='(PCA)')
                
                # 2. Align and merge for plotting
                comparison = pd.DataFrame({
                    'Original': market_values.values,
                    'PCA Fair': pca_fair_values.values
                }, index=[col.replace(f' (Original)', '').replace(f'{derivative_type}: ', '') for col in market_values.index])
                
                if comparison.empty:
                    st.info(f"No {derivative_type} data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                    return
                
                # --- Plot the Derivative ---
                fig, ax = plt.subplots(figsize=(15, 7))
                
                ax.plot(comparison.index, comparison['Original'], label=f'Original Market {derivative_type}', marker='o', linestyle='-', linewidth=2.5, color='blue')
                ax.plot(comparison.index, comparison['PCA Fair'], label=f'PCA Fair {derivative_type} ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                mispricing = comparison['Original'] - comparison['PCA Fair']
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
                
                # Annotate the derivative with the largest absolute mispricing
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    # MODIFIED: * 10000 -> * 100
                    mispricing_value = mispricing.loc[mispricing_contract] * 100 
                    
                    ax.annotate(
                        f"Mispricing: {mispricing_value:.4f} Rate %", # MODIFIED: Unit update
                        (mispricing_contract, comparison.loc[mispricing_contract]['Original']), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )
                
                ax.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type}', fontsize=16)
                ax.set_xlabel(f'{derivative_type} Contract')
                ax.set_ylabel(f'{derivative_type} Value (Price Difference)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle=':', alpha=0.6)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # --- Detailed Table ---
                st.markdown(f"###### {derivative_type} Mispricing")
                detailed_comparison = comparison.copy()
                detailed_comparison.index.name = f'{derivative_type} Contract'
                # MODIFIED: Column name and scaling updated
                detailed_comparison['Mispricing (Rate %)'] = mispricing * 100 
                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': f'Original {derivative_type}', 'PCA Fair': f'PCA Fair {derivative_type}'}
                )
                
                st.dataframe(
                    detailed_comparison.style.format({
                        f'Original {derivative_type}': "{:.4f}",
                        f'PCA Fair {derivative_type}': "{:.4f}",
                        'Mispricing (Rate %)': "{:.4f}" # MODIFIED: Formatting to 4 decimals for clarity
                    }), 
                    use_container_width=True
                )

            except KeyError:
                 st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for {derivative_type}. Please choose a different date within the historical range.")


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

            # --- Plot the Curve ---
            fig_curve, ax_curve = plt.subplots(figsize=(15, 7))
            
            ax_curve.plot(curve_comparison.index, curve_comparison['Original'], label='Original Market Price', marker='o', linestyle='-', linewidth=2.5, color='blue')
            ax_curve.plot(curve_comparison.index, curve_comparison['PCA Fair'], label=f'PCA Fair Price ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
            
            ax_curve.set_title('Market Price Curve vs. PCA Fair Value Curve (Price = 100 - Rate)', fontsize=16)
            ax_curve.set_xlabel('Contract Maturity')
            ax_curve.set_ylabel('Price (100 - Rate)')
            ax_curve.legend(loc='upper right')
            ax_curve.grid(True, linestyle=':', alpha=0.6)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_curve)

            # --- Detailed Contract Price/Rate Table (Outright) ---
            st.markdown("###### Outright Price and Rate Mispricing")
            detailed_comparison = curve_comparison.copy()
            detailed_comparison.index.name = 'Contract'
            detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
            detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
            # MODIFIED: Column name and scaling updated
            detailed_comparison['Mispricing (Rate %)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 100
            
            detailed_comparison = detailed_comparison.rename(
                columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'}
            )
            detailed_comparison = detailed_comparison[[
                'Original Price', 'Original Rate (%)', 'PCA Fair Price', 'PCA Fair Rate (%)', 'Mispricing (Rate %)' # MODIFIED: Column name update
            ]]
            
            st.dataframe(
                detailed_comparison.style.format({
                    'Original Price': "{:.4f}",
                    'PCA Fair Price': "{:.4f}",
                    'Original Rate (%)': "{:.4f}",
                    'PCA Fair Rate (%)': "{:.4f}",
                    'Mispricing (Rate %)': "{:.4f}" # MODIFIED: Formatting to 4 decimals for clarity
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


    else:
        st.error("PCA failed. Please check your data quantity and quality.")
