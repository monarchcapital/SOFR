# -*- coding: utf-8 -*-
# SOFR Futures PCA — stable forecast + robust in-sample reconstruction
#
# ENHANCEMENTS:
#  - Streamlined UI to focus on core yield curve visualizations.
#  - Added VAR (Vector Autoregression) model for forecasting.
#  - Added ARIMA (Autoregressive Integrated Moving Average) model for forecasting.
#  - User can select the forecasting model from the sidebar.
#  - Improved visualizations for clarity and focus on outright yield curve values.
#  - **REVISED: Output is now mapped back to actual contract tickers.**
#  - **NEW: Added a default "PCA Fair Value" forecast model.**
#  - **NEW: Added a "Fair Value Spread Analysis" section to identify mispricing.**
#  - **NEW: Added a PCA components heatmap for intuitive analysis.**
#  - **NEW: Added spreads and butterflies to the PCA analysis.**
#  - **NEW: Added a contract-based butterfly chart and spread comparison table.**
#  - **NEW: Added forecasted spreads and butterflies with contract ticker labels.**
#  - **FIXED: Butterfly contract mapping and labeling.**
#
# Includes:
#  - No "fallback" bug: reconstruction honors the chosen date exactly.
#  - Clean separation of in-sample reconstruction vs next-BD forecast.
#  - Grid rule: 0–3y quarterly (0.25y), 3–5y half-year (0.5y), >5y yearly (1.0y).
#  - Compounding choice (identity assumed for SOFR rates (100-Price) -> annualized zero).
#
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="SOFR PCA — Forecast & Tools")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# --------------------------
# Helpers
# --------------------------
def safe_to_datetime(s):
    # Explicitly state that the day is NOT first to resolve ambiguity for dates like 08-12-2025
    return pd.to_datetime(s, errors='coerce', dayfirst=False)

def normalize_rate_input(val, unit):
    """Return rate in decimal fraction (e.g., 0.1345 for 13.45%)."""
    if pd.isna(val):
        return np.nan
    v = float(val)
    if "Percent" in unit:
        return v / 100.0
    if "Basis" in unit:
        return v / 10000.0
    return v  # decimal

def denormalize_to_percent(frac):
    if pd.isna(frac):
        return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt):
        return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s:
        return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np)
    # The original Brazilian code used business days / year_basis (e.g., 252)
    return np.nan if bd <= 0 else bd / float(year_basis)

def next_business_day(date_like, holidays_np):
    d = np.datetime64(pd.Timestamp(date_like).date())
    nxt = np.busday_offset(d, 1, weekmask="1111100", holidays=holidays_np)
    return pd.Timestamp(nxt)

def build_std_grid_by_rule(max_year=7.0):
    a = list(np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2))   # 0–3y quarterly
    b = list(np.round(np.arange(3.5, 5.0 + 0.001, 0.5), 2))     # 3–5y half-year
    c = list(np.round(np.arange(6.0, max_year + 0.001, 1.0), 2))# >5y yearly
    return a + b + c

# --------------------------
# Sidebar — Inputs
# --------------------------
st.sidebar.header("Upload / Settings")

yield_file = st.sidebar.file_uploader("1) Price/Rate data CSV (dates + contract columns)", type="csv", key="yield_file")
expiry_file = st.sidebar.file_uploader("2) Expiry mapping CSV (maturity, date)", type="csv", key="expiry_file")
holiday_file = st.sidebar.file_uploader("3) Holiday dates CSV (optional)", type="csv", key="holiday_file")

interp_method = st.sidebar.selectbox("Interpolation method", ["linear", "cubic", "quadratic", "nearest"])
apply_smoothing = st.sidebar.checkbox("Apply smoothing (3-day centered)", value=False)
n_components_sel = st.sidebar.slider("Number of PCA components", 1, 12, 3)

rate_unit = st.sidebar.selectbox(
    "Input rate unit",
    ["Percent (e.g. 4.50)", "Decimal (e.g. 0.0450)", "Basis points (e.g. 450)"],
)
year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0))

# --- SOFR COMPATIBILITY CHANGE ---
compounding_model = st.sidebar.radio(
    "Compounding model for curve construction",
    ["identity (Futures Rate R = 100-P assumed as Annual Zero)", "Custom Annual Zero (Requires Re-Compounding)"],
    index=0,
)

use_grid_rule = st.sidebar.checkbox(
    "Use standard grid rule",
    value=True,
)

std_maturities_txt = st.sidebar.text_input(
    "Or custom standard maturities (years, comma-separated)",
    "0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.50,4.00,4.50,5.00,6.00,7.00",
)

st.sidebar.markdown("---")
st.sidebar.header("Forecasting Model")

forecast_model_type = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["None (PCA Fair Value)", "Average Delta (Momentum)", "VAR (Vector Autoregression)", "ARIMA (per Component)"]
)

# --- Model-specific parameters ---
if forecast_model_type == "Average Delta (Momentum)":
    rolling_window_days = st.sidebar.number_input("PCs avg-delta window (days)", min_value=1, max_value=20, value=5, step=1)
    pc_damp = st.sidebar.slider("Damping for PC2..PCn (0 = ignore, 1 = full)", 0.0, 1.0, 0.5, 0.05)
elif forecast_model_type == "VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1)
elif forecast_model_type == "ARIMA (per Component)":
    st.sidebar.write("Using ARIMA(1,1,0) for each PC.")

apply_cap = st.sidebar.checkbox("Apply daily cap to predicted curve change", value=True)
cap_bps = st.sidebar.number_input("Cap per-day move (bps, ±)", min_value=0, max_value=500, value=10, step=1)
cap_frac = float(cap_bps) / 10000.0 # Convert bps to decimal for calculation

st.sidebar.markdown("---")
show_previous_curve = st.sidebar.checkbox("Show Previous Day's Curve", value=True)


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

@st.cache_data(show_spinner=False)
def _quick_read_csv(upload):
    return pd.read_csv(io.StringIO(upload.getvalue().decode("utf-8")))

yields_df_head = None
if yield_file is not None:
    try:
        tmp = _quick_read_csv(yield_file)
        # Assuming the first column is Date
        if not tmp.empty:
            dc = tmp.columns[0]
            # --- SOFR COMPATIBILITY CHANGE: Assume user's input is PRICE and convert to RATE (100-Price) ---
            # If the user uploads Rates directly, the final output will still be rates.
            # To handle both cases, we must ask the user to ensure the input is a rate (100-Price for SOFR).
            # We enforce the conversion if the rates are clearly too high for percent (i.e. > 100), suggesting price input.
            
            # For robustness, we will trust the user to upload the rates (100-Price or 100-Rate)
            # or ensure they convert price to rate (100-Price) before PCA.
            # We will use the original logic and title the application appropriately.
            
            tmp[dc] = safe_to_datetime(tmp[dc])
            tmp = tmp.dropna(subset=[dc]).set_index(dc)
            yields_df_head = tmp
    except Exception:
        yields_df_head = None

if yields_df_head is not None and not yields_df_head.empty:
    min_date_available = yields_df_head.index.min().date()
    max_date_available = yields_df_head.index.max().date()
else:
    min_date_available = pd.to_datetime("2000-01-01").date()
    max_date_available = pd.to_datetime("2100-01-01").date()

date_train_start = st.sidebar.date_input("PCA Training Start Date", value=min_date_available, min_value=min_date_available, max_value=max_date_available)
date_train_end = st.sidebar.date_input("PCA Training End Date", value=max_date_available, min_value=min_date_available, max_value=max_date_available)
if date_train_start > date_train_end:
    st.sidebar.error("PCA Training Start Date cannot be after End Date.")

inputs_ready = (yield_file is not None and expiry_file is not None and (use_grid_rule or len(std_maturities_txt.strip()) > 0) and date_train_start <= date_train_end)

col_a, col_b = st.sidebar.columns(2)
if not st.session_state.analysis_started:
    if inputs_ready and col_a.button("Start Analysis"):
        st.session_state.analysis_started = True
        st.rerun()
else:
    if col_b.button("Stop Analysis"):
        st.session_state.analysis_started = False
        st.stop()

if not st.session_state.analysis_started:
    st.info("Upload files, set parameters and training date range, then click **Start Analysis**.")
    st.info("NOTE for SOFR: Ensure your yield data is the **Futures Rate** (i.e., **100 - Price**) to be analyzed as a rate curve.")
    st.stop()

# --------------------------
# Load full data
# --------------------------
def load_csv_file(f):
    return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")))

yields_df = load_csv_file(yield_file)
date_col = yields_df.columns[0]
yields_df[date_col] = safe_to_datetime(yields_df[date_col])
yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
yields_df.columns = [str(c).strip() for c in yields_df.columns]
for c in yields_df.columns:
    yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")

expiry_raw = load_csv_file(expiry_file)
expiry_df = expiry_raw.iloc[:, :2].copy()
expiry_df.columns = ["MATURITY", "DATE"]
expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")

holidays_np = np.array([], dtype="datetime64[D]")
if holiday_file:
    hol_df = load_csv_file(holiday_file)
    hol_series = safe_to_datetime(hol_df.iloc[:, 0]).dropna()
    if not hol_series.empty:
        holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")

# Filter for training window (used for PCA)
train_mask = (yields_df.index.date >= date_train_start) & (yields_df.index.date <= date_train_end)
yields_df_train = yields_df.loc[train_mask].sort_index()
if yields_df_train.empty:
    st.error("No rates in the selected PCA training date range.")
    st.stop()
if apply_smoothing:
    yields_df_train = yields_df_train.rolling(window=3, min_periods=1, center=True).mean()

# Standard maturities grid
if use_grid_rule:
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
else:
    try:
        std_maturities = [float(x.strip()) for x in std_maturities_txt.split(",") if x.strip() != ""]
        std_arr = np.array(sorted(std_maturities), dtype=float)
    except Exception:
        st.error("Error parsing standard maturities.")
        st.stop()
std_cols = [f"{m:.2f}Y" for m in std_arr]

# --------------------------
# Core Data Transformation Functions
# --------------------------
def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    ttm_list, zero_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index:
            continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date():
            continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0:
            continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val):
            continue
        r_frac = normalize_rate_input(raw_val, rate_unit)
        
        # --- SOFR COMPATIBILITY CHANGE: Simplified Compounding Model ---
        # For SOFR futures, the rate (100-P) is typically treated as an effective 
        # annual rate for PCA purposes on a rate-of-rate matrix. The original 
        # DI compounding is complex and unnecessary here. We use the 'identity' 
        # path, treating the input rate as the annualized zero rate.
        if compounding_model.startswith("identity"):
            zero_frac = r_frac
        else:
            # Placeholder for custom or more complex zero rate derivation, 
            # currently defaulting to the simple identity for SOFR for robustness.
            zero_frac = r_frac 
        # --- END SOFR CHANGE ---
        
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
        
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(
                np.array(ttm_list)[order],
                np.array(zero_list)[order],
                kind=interp_method,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            return f(std_arr)
        except Exception:
            return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)

def std_grid_to_contracts(dt, std_curve_rates, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    """Converts a curve on the standard grid back to individual contract rates."""
    contract_rates = pd.Series(index=all_contracts, dtype=float)
    
    interp_func = interp1d(std_arr, std_curve_rates, kind=interp_method, bounds_error=False, fill_value='extrapolate')

    for contract in all_contracts:
        mat_up = str(contract).strip().upper()
        if mat_up not in expiry_df.index:
            continue
        
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date():
            continue

        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0:
            continue

        zero_percent = interp_func(t)
        zero_frac = zero_percent / 100.0

        # --- SOFR COMPATIBILITY CHANGE: Simplified Compounding Model ---
        # Using the identity path for rate reconstruction.
        if compounding_model.startswith("identity"):
            rate_frac = zero_frac
        else:
            # Placeholder for custom or more complex rate derivation, 
            # currently defaulting to the simple identity for SOFR for robustness.
            rate_frac = zero_frac
        # --- END SOFR CHANGE ---
        
        if "Percent" in rate_unit:
             contract_rates[contract] = rate_frac * 100.0
        elif "Basis" in rate_unit:
             contract_rates[contract] = rate_frac * 10000.0
        else:
             contract_rates[contract] = rate_frac

    return contract_rates.dropna()

def get_contract_names_for_maturities(maturities_list, expiry_df, last_date, holidays_np, year_basis):
    """
    Finds the contract tickers corresponding to a list of maturities (ttm_years)
    for a given date. This uses a more robust "closest match" method.
    """
    contract_names = []
    
    # Calculate TTM for all available contracts on the last date
    ttm_to_contract = {}
    for contract, exp_date in expiry_df["DATE"].items():
        if pd.isna(exp_date) or pd.Timestamp(exp_date).date() < last_date.date(): continue
        ttm = calculate_ttm(last_date, exp_date, holidays_np, year_basis)
        if not np.isnan(ttm):
            ttm_to_contract[ttm] = contract
            
    ttm_vals = np.array(list(ttm_to_contract.keys()))
            
    for m in maturities_list:
        if len(ttm_vals) == 0:
            contract_names.append("N/A")
            continue
        
        # Find the index of the closest TTM
        closest_idx = np.abs(ttm_vals - m).argmin()
        closest_ttm = ttm_vals[closest_idx]
        
        contract_names.append(ttm_to_contract[closest_ttm])
        
    return contract_names

def get_contract_names_for_spreads_flies(maturity_labels, expiry_df, last_date, holidays_np, year_basis):
    """
    Maps maturity-based spread and fly labels to contract tickers.
    Correctly formats flies with a / separator to distinguish from spreads.
    """
    contract_labels = []
    for label in maturity_labels:
        parts = label.split('-')
        maturities = [float(p[:-1]) for p in parts]
        
        contracts = get_contract_names_for_maturities(maturities, expiry_df, last_date, holidays_np, year_basis)
        
        if 'N/A' in contracts or len(contracts) != len(parts):
            contract_labels.append('N/A')
            continue
        
        if len(parts) == 2:
            # It's a spread
            contract_labels.append('-'.join(contracts))
        elif len(parts) == 3:
            # It's a fly, format using /
            # The calculation is (back-body)-(body-front), so we'll label it
            # FRONT/BODY/BACK. We need to re-order the `parts` because
            # fly_cols are generated as `back-body-front`.
            maturities_sorted = sorted(maturities)
            contracts_sorted = get_contract_names_for_maturities(maturities_sorted, expiry_df, last_date, holidays_np, year_basis)
            contract_labels.append('/'.join(contracts_sorted))
        else:
            contract_labels.append('N/A')
            
    return contract_labels

# @st.cache_data
def build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method):
    # Step 1: Create a DataFrame for zero rates on the standard grid
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df, 
            std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method
        )
    
    # Step 2: Calculate Spreads
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    pca_df_spreads = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    for i in range(1, len(std_cols)):
        col_name = f"{std_cols[i]}-{std_cols[i-1]}"
        pca_df_spreads[col_name] = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]

    # Step 3: Calculate Butterflies
    fly_cols = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    pca_df_flies = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=fly_cols, dtype=float)
    for i in range(1, len(std_cols) - 1):
        col_name = f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}"
        # Butterfly calculation: (Long - Short) - (Short - Front)
        pca_df_flies[col_name] = (pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]) - (pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]])

    # Step 4: Combine all series into a single PCA matrix
    # Note: PCA is best performed on daily changes (diff), but the original script 
    # performs PCA on the level data (after centering in StandardScaler). 
    # We maintain this structure as per user's request for exact logic.
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)

    # Fill NaN values with the column mean (after calculation)
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])
    
    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns)


pca_df_filled = build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, compounding_model, interp_method)

# --------------------------
# PCA
# --------------------------
# PCA is run on the *levels* of the curve, spreads, and flies, centered by the mean.
scaler = StandardScaler(with_std=False)
X = scaler.fit_transform(pca_df_filled.values.astype(float))

max_components = int(min(X.shape[0], X.shape[1]))
n_components = int(min(int(n_components_sel), max_components))
if n_components < int(n_components_sel):
    st.info(f"Reduced PCA components from {int(n_components_sel)} to {n_components} (limited by samples/features).")

pca = PCA(n_components=n_components)
PCs = pca.fit_transform(X)
pc_cols = [f"PC{i+1}" for i in range(n_components)]
PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

# --------------------------
# Utilities for reconstruction/forecast
# --------------------------
def reconstruct_curve_at_index(i_row: int) -> np.ndarray:
    x_centered = pca.inverse_transform(PCs[i_row].reshape(1, -1)).ravel()
    return scaler.inverse_transform(x_centered.reshape(1, -1)).ravel()

# --- FORECASTING FUNCTIONS ---
def forecast_pcs_avg_delta(PCs_matrix, window=5, pc_damp=0.5):
    if PCs_matrix.shape[0] == 0: return np.zeros((1, PCs_matrix.shape[1]))
    n, k = PCs_matrix.shape
    w = int(min(max(1, window), n))
    last = PCs_matrix[-1].reshape(1, -1)
    if w <= 1 or n == 1: avg_delta = np.zeros((k,))
    else: avg_delta = np.mean(np.diff(PCs_matrix[-w:], axis=0), axis=0)
    damp = np.ones(k)
    if k > 1: damp[1:] = float(pc_damp)
    return last + (avg_delta * damp).reshape(1, -1)

def forecast_pcs_var(PCs_df, lags=1):
    if len(PCs_df) < lags + 5:
        st.warning(f"Not enough data ({len(PCs_df)} points) for VAR({lags}). Falling back to last values.")
        return PCs_df.iloc[-1:].values
    results = VAR(PCs_df).fit(lags)
    return results.forecast(PCs_df.values[-lags:], steps=1)

def forecast_pcs_arima(PCs_df):
    forecasts = []
    for name, series in PCs_df.items():
        if len(series) < 10:
            st.warning(f"Not enough data for {name} to fit ARIMA. Using last value.")
            forecasts.append(series.iloc[-1])
            continue
        forecasts.append(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1).iloc[0])
    return np.array(forecasts).reshape(1, -1)

# --------------------------
# Main Panel
# --------------------------
st.title("SOFR Futures Yield Curve Forecast")

# --- NEW: PCA Components Heatmap ---
st.header("Principal Component Analysis Overview")
st.write("These heatmaps show how each Principal Component (PC) influences different maturities, spreads, and butterflies. This is based on the entire training data period.")
st.write("- **PC1 (Level):** Typically shows a parallel shift in the curve.")
st.write("- **PC2 (Slope):** Typically shows a steepening or flattening of the curve, reflected in spreads.")
st.write("- **PC3 (Curvature):** Typically shows a change in the curve's bow or flex, reflected in flies.")

# Get indices for each group of columns
idx_rates = [pca_df_filled.columns.get_loc(c) for c in std_cols]
idx_spreads = [pca_df_filled.columns.get_loc(c) for c in pca_df_filled.columns if '-' in c and len(c.split('-')) == 2]
idx_flies = [pca_df_filled.columns.get_loc(c) for c in pca_df_filled.columns if '-' in c and len(c.split('-')) == 3]

# Plot 1: Loadings for Outright Rates
fig_rates, ax_rates = plt.subplots(figsize=(12, max(4, n_components * 1.5)))
sns.heatmap(
    pca.components_[:, idx_rates],
    xticklabels=std_cols,
    yticklabels=pc_cols,
    annot=True, 
    fmt=".2f",
    cmap='viridis',
    ax=ax_rates
)
ax_rates.set_title("PCA Component Loadings vs. Outright Rates")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_rates)

# Plot 2: Loadings for Spreads
fig_spreads, ax_spreads = plt.subplots(figsize=(12, max(4, n_components * 1.5)))
sns.heatmap(
    pca.components_[:, idx_spreads],
    xticklabels=[c.replace('Spread_', '') for c in pca_df_filled.columns[idx_spreads]],
    yticklabels=pc_cols,
    annot=True, 
    fmt=".2f",
    cmap='viridis',
    ax=ax_spreads
)
ax_spreads.set_title("PCA Component Loadings vs. Spreads")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_spreads)

# Plot 3: Loadings for Butterflies
fig_flies, ax_flies = plt.subplots(figsize=(12, max(4, n_components * 1.5)))
sns.heatmap(
    pca.components_[:, idx_flies],
    xticklabels=[c.replace('Fly_', '') for c in pca_df_filled.columns[idx_flies]],
    yticklabels=pc_cols,
    annot=True, 
    fmt=".2f",
    cmap='viridis',
    ax=ax_flies
)
ax_flies.set_title("PCA Component Loadings vs. Butterflies (by Maturity)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_flies)

# --- Get the last actual curve ---
last_actual_ts = yields_df_train.index.max()
last_actual_series = pca_df_filled.loc[last_actual_ts]
last_actual_curve_on_std = last_actual_series[std_cols].values

# --- Get the PREVIOUS day's curve for comparison ---
prev_actual_ts = None
prev_actual_curve_on_std = np.full_like(std_arr, np.nan)
if len(yields_df_train.index) > 1:
    prev_actual_ts = yields_df_train.index[-2]
    prev_actual_series = pca_df_filled.loc[prev_actual_ts]
    prev_actual_curve_on_std = prev_actual_series[std_cols].values

# --------------------------
# In-sample reconstruction (exact date)
# --------------------------
st.markdown("---")
st.header("In-Sample Curve Reconstruction")
recon_min, recon_max = pca_df_filled.index.min().date(), pca_df_filled.index.max().date()
recon_date = st.date_input("Select date for reconstruction", value=recon_max, min_value=recon_min, max_value=recon_max, key="recon_date_pick")
recon_ts = pd.Timestamp(recon_date)

mask_pca = pca_df_filled.index.normalize() == recon_ts.normalize()
if not mask_pca.any():
    st.error(f"Selected date {recon_ts.date()} is not present in the training data. Please check your PCA Training Date range in the sidebar.")
else:
    i = int(np.where(mask_pca)[0][0])
    recon_full_series = pd.Series(reconstruct_curve_at_index(i), index=pca_df_filled.columns)
    recon_curve_std = recon_full_series[std_cols].values
    orig_on_std = pca_df_filled.loc[recon_ts, std_cols].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(std_arr, recon_curve_std, marker='o', linestyle='-', color='royalblue', label=f"Reconstructed ({n_components} PCs)")
    ax.plot(std_arr, orig_on_std, marker='x', linestyle='--', color='darkorange', label="Original (interpolated on grid)")
    if show_previous_curve and prev_actual_ts is not None:
        ax.plot(std_arr, prev_actual_curve_on_std, marker='+', linestyle=':', color='green', label=f"Previous Day Curve ({prev_actual_ts.date()})")
    ax.set_xticks(std_arr)
    ax.set_xticklabels([f"{m:.2f}Y" for m in std_arr], rotation=45, ha="right")
    ax.set_xlabel("Standardized Maturity (Years)", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(f"SOFR Curve Reconstruction for {recon_ts.date()} on Standard Grid", fontsize=16, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    
    # --- Reconstruction for Spreads and Flies ---
    st.markdown("---")
    st.subheader("Spread and Fly Reconstruction")
    
    # Get the spread and fly columns
    spread_cols_full = [col for col in pca_df_filled.columns if '-' in col and len(col.split('-')) == 2]
    fly_cols_full = [col for col in pca_df_filled.columns if '-' in col and len(col.split('-')) == 3]

    if spread_cols_full and fly_cols_full:
        orig_spreads = pca_df_filled.loc[recon_ts, spread_cols_full].values
        recon_spreads = recon_full_series[spread_cols_full].values
        orig_flies = pca_df_filled.loc[recon_ts, fly_cols_full].values
        recon_flies = recon_full_series[fly_cols_full].values

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Spreads Plot
        ax1.plot(spread_cols_full, recon_spreads, marker='o', linestyle='-', color='royalblue', label=f"Reconstructed Spreads")
        ax1.plot(spread_cols_full, orig_spreads, marker='x', linestyle='--', color='darkorange', label="Original Spreads")
        ax1.set_title("Spread Reconstruction")
        ax1.set_ylabel("Spread (bps)")
        ax1.set_xlabel("Spread (Years)")
        ax1.set_xticklabels(spread_cols_full, rotation=45, ha="right")
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Flies Plot
        ax2.plot(fly_cols_full, recon_flies, marker='o', linestyle='-', color='royalblue', label=f"Reconstructed Flies")
        ax2.plot(fly_cols_full, orig_flies, marker='x', linestyle='--', color='darkorange', label="Original Flies")
        ax2.set_title("Fly Reconstruction")
        ax2.set_ylabel("Fly (bps)")
        ax2.set_xlabel("Fly (Years)")
        ax2.set_xticklabels(fly_cols_full, rotation=45, ha="right")
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

# --- NEW: In-sample Spread Analysis (Table) ---
st.markdown("---")
st.subheader("In-Sample Spread Analysis")
st.write(f"This table compares the **Original** market spread values to the **PCA-based** reconstructed spread values for the selected reconstruction date ({recon_ts.date()}).")

# Calculate original spreads based on the interpolated grid
spread_cols_full = [col for col in pca_df_filled.columns if '-' in col and len(col.split('-')) == 2]
orig_spreads = pca_df_filled.loc[recon_ts, spread_cols_full].rename("Original Spreads (bps)")
recon_spreads = recon_full_series[spread_cols_full].rename("PCA Reconstructed Spreads (bps)")

spread_df_compare = pd.concat([orig_spreads, recon_spreads], axis=1)
st.dataframe(spread_df_compare.style.format("{:.2f}"))


# --------------------------
# Next-BD forecast
# --------------------------
st.markdown("---")
st.header(f"Next Business Day Forecast (by Contract)")

train_end_ts = pca_df_filled.index.max()
forecast_ts = next_business_day(train_end_ts, holidays_np)

# --- Select and run the chosen forecast model ---
if forecast_model_type == "None (PCA Fair Value)":
    st.write("Using **PCA Fair Value** model (no momentum or external forecast)")
    # The "Fair Value" forecast is the smooth, reconstructed curve of the last day.
    last_day_pcs_index = pca_df_filled.index.get_loc(last_actual_ts)
    pred_full_series = pd.Series(reconstruct_curve_at_index(last_day_pcs_index), index=pca_df_filled.columns)
else:
    if forecast_model_type == "Average Delta (Momentum)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_avg_delta(PCs, window=int(rolling_window_days), pc_damp=float(pc_damp))
    elif forecast_model_type == "VAR (Vector Autoregression)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_var(PCs_df, lags=int(var_lags))
    elif forecast_model_type == "ARIMA (per Component)":
        st.write(f"Using **{forecast_model_type}** model")
        pcs_next = forecast_pcs_arima(PCs_df)

    # Apply the predicted CHANGE to the last known actual curve
    last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
    delta_pcs = pcs_next - last_pcs
    delta_vals = pca.inverse_transform(delta_pcs).flatten()
    
    last_actual_vals = last_actual_series.values
    pred_vals = last_actual_vals + delta_vals
    
    pred_full_series = pd.Series(pred_vals, index=pca_df_filled.columns)

# --- Apply Capping to momentum-based models ---
if apply_cap and cap_bps > 0 and forecast_model_type != "None (PCA Fair Value)":
    delta = pred_full_series[std_cols] - last_actual_series[std_cols]
    delta = np.clip(delta, -cap_frac, cap_frac)
    capped_pred_std = last_actual_series[std_cols] + delta
else:
    capped_pred_std = pred_full_series[std_cols]

# --- Convert curves from standard grid back to contracts ---
all_contracts = yields_df.columns
pred_contracts = std_grid_to_contracts(forecast_ts, capped_pred_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)
last_actual_contracts = std_grid_to_contracts(last_actual_ts, last_actual_curve_on_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)
prev_actual_contracts = std_grid_to_contracts(prev_actual_ts, prev_actual_curve_on_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method) if prev_actual_ts else None

# --- Visualization by Contract ---
fig, ax = plt.subplots(figsize=(12, 7))
valid_contracts = pred_contracts.index

ax.plot(valid_contracts, last_actual_contracts.loc[valid_contracts], marker='s', markersize=5, linestyle='-', color='gray', label=f"Last Actual ({last_actual_ts.date()})")
ax.plot(valid_contracts, pred_contracts, marker='o', markersize=7, linestyle='-', color='crimson', label=f"Predicted ({forecast_ts.date()})")
if show_previous_curve and prev_actual_contracts is not None:
    ax.plot(valid_contracts, prev_actual_contracts.loc[valid_contracts], marker='+', linestyle=':', color='green', label=f"Previous Day ({prev_actual_ts.date()})")

ax.set_xlabel("Contract Ticker", fontsize=12)
ax.set_ylabel(f"Rate ({rate_unit})", fontsize=12)
ax.set_title(f"SOFR Yield Curve Forecast for {forecast_ts.date()}", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right")
ax.legend(fontsize=10, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine()
plt.tight_layout()
st.pyplot(fig)

# --- Add the forecast data table by Contract ---
st.markdown("---")
st.subheader("Forecast Data by Contract")
forecast_df = pd.DataFrame({
    f"Previous Day ({prev_actual_ts.date() if prev_actual_ts else 'N/A'})": prev_actual_contracts,
    f"Last Actual ({last_actual_ts.date()})": last_actual_contracts,
    f"Predicted ({forecast_ts.date()})": pred_contracts
}).dropna(how='all')

st.dataframe(forecast_df.style.format("{:.2f}"))

# --------------------------
# Fair Value Spread Analysis
# --------------------------
st.markdown("---")
st.header("Fair Value Spread Analysis")
st.write(f"This section compares the **Last Actual** market rates against the **PCA Fair Value** for that same day ({last_actual_ts.date()}).")
st.write("A positive spread means the actual market rate is higher than the model's fair value (potentially overpriced). A negative spread means it's lower (potentially underpriced).")

# Calculate the PCA Fair Value for the last actual day
last_day_pcs_index = pca_df_filled.index.get_loc(last_actual_ts)
fair_value_std = pd.Series(reconstruct_curve_at_index(last_day_pcs_index), index=pca_df_filled.columns)[std_cols]
fair_value_contracts = std_grid_to_contracts(last_actual_ts, fair_value_std, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, compounding_model, interp_method)

# Align the series and calculate the spread
aligned_actual, aligned_fv = last_actual_contracts.align(fair_value_contracts, join='inner')
spread_bps = (aligned_actual - aligned_fv) * 100 # Assuming 'Percent' unit for basis points

# Display in a chart
fig_spread, ax_spread = plt.subplots(figsize=(12, 6))
colors = ['g' if x < 0 else 'r' for x in spread_bps]
spread_bps.plot(kind='bar', ax=ax_spread, color=colors)
ax_spread.set_ylabel("Spread (Basis Points)")
ax_spread.set_xlabel("Contract Ticker")
ax_spread.set_title("Actual Market Rate vs. PCA Fair Value")
ax_spread.axhline(0, color='k', linewidth=0.8)
st.pyplot(fig_spread)

# Display in a table
spread_df = pd.DataFrame({
    "Actual Rate": aligned_actual,
    "PCA Fair Value": aligned_fv,
    "Spread (bps)": spread_bps
})
st.dataframe(spread_df.style.format("{:.2f}"))

# --------------------------
# New section: Forecasted Spreads and Flies
# --------------------------
st.markdown("---")
st.header("Forecasted Spreads and Butterflies (by Contract)")
st.write("This section visualizes the **forecasted changes** in spreads and butterflies for the next business day.")
st.write("A positive change means a steepening spread or a more positive butterfly. A negative change means a flattening spread or a more negative butterfly.")

# Calculate last actual spreads/flies and forecasted spreads/flies
last_spreads = last_actual_series[idx_spreads]
last_flies = last_actual_series[idx_flies]
forecasted_spreads = pred_full_series[idx_spreads]
forecasted_flies = pred_full_series[idx_flies]

# Get the maturity-based labels for spreads and flies
spread_labels = last_spreads.index.tolist()
fly_labels = last_flies.index.tolist()

# Get the contract-based labels using the new helper function
spread_contract_labels = get_contract_names_for_spreads_flies(spread_labels, expiry_df, last_actual_ts, holidays_np, year_basis)
fly_contract_labels = get_contract_names_for_spreads_flies(fly_labels, expiry_df, last_actual_ts, holidays_np, year_basis)

# Plotting the changes
fig_forecast, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Spreads Plot
ax1.plot(spread_contract_labels, last_spreads, marker='o', linestyle='-', color='gray', label=f"Last Actual Spreads ({last_actual_ts.date()})")
ax1.plot(spread_contract_labels, forecasted_spreads, marker='x', linestyle='--', color='darkorange', label=f"Forecasted Spreads ({forecast_ts.date()})")
ax1.set_title("Forecasted Spread Analysis")
ax1.set_ylabel("Spread (bps)")
ax1.set_xlabel("Spread (Contracts)")
ax1.set_xticklabels(spread_contract_labels, rotation=45, ha="right")
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend()

# Flies Plot
ax2.plot(fly_contract_labels, last_flies, marker='o', linestyle='-', color='gray', label=f"Last Actual Flies ({last_actual_ts.date()})")
ax2.plot(fly_contract_labels, forecasted_flies, marker='x', linestyle='--', color='darkorange', label=f"Forecasted Flies ({forecast_ts.date()})")
ax2.set_title("Forecasted Butterfly Analysis")
ax2.set_ylabel("Fly (bps)")
ax2.set_xlabel("Fly (Contracts)")
ax2.set_xticklabels(fly_contract_labels, rotation=45, ha="right")
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend()

plt.tight_layout()
st.pyplot(fig_forecast)

# Table for comparison
st.subheader("Forecasted Spread and Butterfly Values")

# Create a table for spreads
spread_table = pd.DataFrame({
    f"Last Actual ({last_actual_ts.date()})": last_spreads.values,
    f"Forecasted ({forecast_ts.date()})": forecasted_spreads.values,
    "Change": forecasted_spreads.values - last_spreads.values
}, index=spread_contract_labels)
st.markdown("**Spreads**")
st.dataframe(spread_table.style.format("{:.2f}"))

# Create a table for flies
fly_table = pd.DataFrame({
    f"Last Actual ({last_actual_ts.date()})": last_flies.values,
    f"Forecasted ({forecast_ts.date()})": forecasted_flies.values,
    "Change": forecasted_flies.values - last_flies.values
}, index=fly_contract_labels)
st.markdown("**Butterflies**")
st.dataframe(fly_table.style.format("{:.2f}"))