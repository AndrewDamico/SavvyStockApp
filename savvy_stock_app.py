import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Savvy Stock Portfolio Optimizer", layout="wide")
st.title("📈 Savvy Stock Selection: Portfolio Optimizer")

st.markdown("Edit the stock parameters below to run custom portfolio scenarios:")

# Initial stock data
initial_data = pd.DataFrame({
    "Stock": ["BB", "LOP", "ILI", "HEAL", "QUI", "AUA"],
    "Start Price": [60, 127, 4, 50, 150, 20],
    "Expected Price": [72, 127 * 1.42, 8, 75, 150 * 1.46, 26],
    "Variance": [0.032, 0.1, 0.333, 0.125, 0.065, 0.08],
})

# Editable table
stock_df = st.data_editor(initial_data, num_rows="fixed", use_container_width=True)

# Enforce numeric types
stock_df["Start Price"] = pd.to_numeric(stock_df["Start Price"], errors="coerce")
stock_df["Expected Price"] = pd.to_numeric(stock_df["Expected Price"], errors="coerce")
stock_df["Variance"] = pd.to_numeric(stock_df["Variance"], errors="coerce")

# Drop any invalid rows
invalid_rows = stock_df[stock_df.isnull().any(axis=1)]
stock_df = stock_df.dropna()

# Calculate expected returns
expected_returns = ((stock_df["Expected Price"] - stock_df["Start Price"]) / stock_df["Start Price"]).to_numpy(dtype=np.float64)
asset_names = stock_df["Stock"].tolist()

# Covariance matrix
base_corr = np.array([
    [1, 0.1, 0.8, -0.9, -0.8, 0.4],
    [0.1, 1, -0.7, -0.5, 0.2, 0],
    [0.8, -0.7, 1, -0.6, -0.3, 0.5],
    [-0.9, -0.5, -0.6, 1, 0.6, -0.7],
    [-0.8, 0.2, -0.3, 0.6, 1, -0.3],
    [0.4, 0, 0.5, -0.7, -0.3, 1]
], dtype=np.float64)

n_assets = len(expected_returns)
base_corr = base_corr[:n_assets, :n_assets]
variances = stock_df["Variance"].to_numpy(dtype=np.float64)
std_devs = np.sqrt(variances)
cov_matrix = np.outer(std_devs, std_devs) * base_corr
cov_matrix = (cov_matrix + cov_matrix.T) / 2
eigvals = np.linalg.eigvalsh(cov_matrix)
if np.any(eigvals < 0):
    cov_matrix += np.eye(n_assets) * (abs(min(eigvals)) + 1e-5)

# Sidebar controls
st.sidebar.header("Controls")
min_return = st.sidebar.slider("Minimum Expected Return", 0.05, 0.60, 0.10, 0.01)
max_alloc_toggle = st.sidebar.checkbox("Enable Max Allocation Constraint?", value=True)
max_alloc_value = st.sidebar.slider("Max Allocation Per Stock (%)", 10, 100, 100, 5) / 100 if max_alloc_toggle else None
highlight_mvp = st.sidebar.checkbox("Highlight Minimum Variance Portfolio", value=True)
solver = st.sidebar.selectbox("Solver", options=["ECOS", "SCS", "OSQP"])

# Efficient frontier setup
target_returns = np.linspace(0.05, 0.60, 200)
risks, solutions, failed_targets = [], [], []

x = cp.Variable(n_assets, name="weights")
constraints = [cp.sum(x) == 1, x >= 0]
if max_alloc_toggle:
    constraints.append(x <= max_alloc_value)

for target in target_returns:
    prob_constraints = constraints + [expected_returns @ x >= target]
    objective = cp.Minimize(cp.quad_form(x, cov_matrix))
    problem = cp.Problem(objective, prob_constraints)
    try:
        problem.solve(solver=solver)
        if x.value is not None:
            risks.append(np.sqrt(float(x.value.T @ cov_matrix @ x.value)))
            solutions.append(x.value)
        else:
            risks.append(None)
            solutions.append([None] * n_assets)
            failed_targets.append(round(target, 4))
    except Exception:
        risks.append(None)
        solutions.append([None] * n_assets)
        failed_targets.append(round(target, 4))

# Output data
df = pd.DataFrame(solutions, columns=asset_names)
df["Expected Return"] = target_returns
df["Risk (Std Dev)"] = risks
df_valid = df.dropna()

# Display optimal portfolio
if not df_valid.empty and not df_valid[df_valid["Expected Return"] >= min_return].empty:
    selected_row = df_valid[df_valid["Expected Return"] >= min_return].iloc[0]
    st.subheader("Optimal Portfolio at Selected Minimum Return")
    st.write("📌 Expected Return:", round(selected_row["Expected Return"], 3))
    st.write("📉 Risk (Std Dev):", round(selected_row["Risk (Std Dev)"], 3))
    st.dataframe(selected_row[asset_names].T.rename("Weight (%)") * 100)
else:
    st.warning("⚠️ No feasible portfolio found. Try reducing the target return or increasing max allocation.")

# Visualization
st.subheader("📊 Visualization")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📉 Efficient Frontier")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(df["Risk (Std Dev)"], df["Expected Return"], label="Efficient Frontier", color="blue")
    if highlight_mvp and not df_valid.empty:
        min_risk_idx = df_valid["Risk (Std Dev)"].idxmin()
        min_point = df_valid.loc[min_risk_idx]
        ax2.scatter(min_point["Risk (Std Dev)"], min_point["Expected Return"], color='red', zorder=5)
        ax2.annotate("Min Variance Portfolio", xy=(min_point["Risk (Std Dev)"], min_point["Expected Return"]),
                     xytext=(min_point["Risk (Std Dev)"] + 0.01, min_point["Expected Return"]),
                     arrowprops=dict(facecolor='red', arrowstyle="->"))
    ax2.set_xlabel("Risk (Std Dev)")
    ax2.set_ylabel("Expected Return")
    ax2.grid(True)
    st.pyplot(fig2)

with col2:
    st.markdown("### 🧮 Asset Weights vs Return Target")
    fig, ax = plt.subplots(figsize=(6, 5))
    for col in asset_names:
        ax.plot(df["Expected Return"], df[col], label=col)
    ax.set_xlabel("Expected Return")
    ax.set_ylabel("Portfolio Weight")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Diagnostics
st.subheader("📋 Diagnostics")
if not invalid_rows.empty:
    st.error(f"❌ Invalid rows removed from editable table (e.g. missing values):")
    st.dataframe(invalid_rows)
st.info(f"✅ Feasible portfolios computed: {len(df_valid)} / {len(target_returns)}")
if failed_targets:
    st.warning(f"⚠️ Infeasible targets: {len(failed_targets)}. First few: {failed_targets[:5]}...")