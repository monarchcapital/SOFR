import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data ---
# Load the SOFR rates data. 'Date' will be set as the index later.
try:
    # Assuming the files are accessible through the uploaded file names
    df_sofr = pd.read_csv('sofr rates.csv', index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("Error: 'sofr rates.csv' not found. Please ensure the file is correctly uploaded.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---

# The first few rows have many NaN values. We will drop any column that is entirely NaN,
# and then fill remaining NaNs using the 'ffill' (forward fill) method.
# Since the initial rows of the provided snippet are empty, we might need to fill them
# or drop them entirely if they contain no data.
# For simplicity and to use as much data as possible, we will forward fill (ffill)
# and then backward fill (bfill) any remaining NaNs.

print("Original shape:", df_sofr.shape)

# Drop columns that are completely empty (if any exist beyond the snippet view)
df_sofr.dropna(axis=1, how='all', inplace=True)

# Fill forward: fills NaNs with the previous day's rate (common practice for time series)
df_sofr.fillna(method='ffill', inplace=True)

# Fill backward: fills any remaining leading NaNs with the next valid observation
df_sofr.fillna(method='bfill', inplace=True)

# Drop any rows that still contain NaNs (if the entire time series is empty)
df_sofr.dropna(axis=0, how='any', inplace=True)

print("Cleaned shape (after filling NaNs):", df_sofr.shape)

if df_sofr.empty:
    print("Error: DataFrame is empty after cleaning. Check the data content.")
    exit()

# --- 3. Calculate Daily Rate Changes (Returns) ---

# PCA is most effective on changes (returns) rather than absolute levels, 
# as rates are highly correlated.
df_changes = df_sofr.diff().dropna()
print("Changes shape:", df_changes.shape)

if df_changes.empty:
    print("Error: Changes DataFrame is empty. Not enough data points to calculate changes.")
    exit()

X = df_changes.values

# --- 4. Standardize the Data ---

# Standardizing the data is crucial for PCA to ensure all variables 
# contribute equally to the variance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Perform PCA ---

# Instantiate PCA. We choose to keep all components (n_components=None) 
# to calculate the total variance explained.
pca = PCA(n_components=None)
pca.fit(X_scaled)

# --- 6. Results Analysis ---

# Explained variance ratio (eigenvalues)
explained_variance_ratio = pca.explained_variance_ratio_

print("\n--- PCA Results (Fixed Income Risk Factors) ---")
print("\nExplained Variance Ratio by Component:")
for i, ratio in enumerate(explained_variance_ratio[:5]):
    print(f"Component {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"\nCumulative Variance Explained by 1 Component: {cumulative_variance[0]*100:.2f}%")
print(f"Cumulative Variance Explained by 2 Components: {cumulative_variance[1]*100:.2f}%")
print(f"Cumulative Variance Explained by 3 Components: {cumulative_variance[2]*100:.2f}%")
print("These top 3 components typically represent Level, Slope, and Curvature.")


# Principal Components (Loadings/Eigenvectors)
# The sign of the components can be arbitrary, focus on the relative magnitude.
components = pca.components_
component_labels = [f"PC{i+1}" for i in range(components.shape[0])]
maturity_labels = df_sofr.columns

df_components = pd.DataFrame(components, columns=maturity_labels, index=component_labels)

# Display the loadings for the first three components (Level, Slope, Curvature)
print("\nPrincipal Component Loadings (Eigenvectors) - Top 3:")
print(df_components.iloc[:3].to_markdown())

# --- Plotting the Cumulative Explained Variance ---

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='purple')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.legend()

# Save the plot as a base64 image (since we can't save files directly)
# This part is for visualization if the environment supports it
import base64
from io import BytesIO
buffer = BytesIO()
plt.savefig(buffer, format='png')
plt.close(plt.gcf()) # Close the figure to free memory
plot_data = base64.b64encode(buffer.getvalue()).decode()
# Print a placeholder for the plot data
print("\n[Base64 Plot Data for Cumulative Explained Variance]")
# The actual plot will be rendered by the environment using this data.

# --- Interpretive Guidance ---
print("\n--- Interpretation Guidance ---")
print("Component 1 (Level): Look for all positive loadings across all maturities. This is the parallel shift of the yield curve.")
print("Component 2 (Slope): Look for large positive loadings at short maturities and large negative loadings at long maturities (or vice-versa). This is the steepening/flattening of the yield curve.")
print("Component 3 (Curvature): Look for opposite signs in the middle of the curve compared to the ends. This controls the 'hump' of the curve.")
