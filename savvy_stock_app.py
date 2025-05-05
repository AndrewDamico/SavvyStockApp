import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cvxpy as cp

st.set_page_config(page_title="Savvy Stock Portfolio Optimizer with Risk Preferences", layout="wide")
st.title("📈 Savvy Stock Portfolio Optimizer (Risk Aversion Adjustable)")

st.markdown("Explore how changing your return expectations and risk aversion affects your optimal portfolio.")

# Editable stock table
initial_data = pd.DataFrame({
    "Stock": ["BB", "LOP", "ILI", "HEAL", "QUI", "AUA"],
    "Start Price": [60, 127, 4, 50, 150, 20],
    "Expected Price": [72, 127 * 1.42, 8, 75, 150 * 1.46, 26],
    "Variance": [0.032, 0.1, 0.333, 0.125, 0.065, 0.08],
})
stock_df = st.data_editor(initial_data, num_rows="fixed", use_container_width=True)
stock_df = stock_df.dropna()
stock_df["Start Price"] = pd.to_numeric(stock_df["Start Price"], errors="coerce")
stock_df["Expected Price"] = pd.to_numeric(stock_df["Expected Price"], errors="coerce")
stock_df["Variance"] = pd.to_numeric(stock_df["Variance"], errors="coerce")
expected_returns = ((stock_df["Expected Price"] - stock_df["Start Price"]) / stock_df["Start Price"]).to_numpy()
asset_names = stock_df["Stock"].tolist()

# Covariance matrix
base_corr = np.array([
    [1, 0.1, 0.8, -0.9, -0.8, 0.4],
    [0.1, 1, -0.7, -0.5, 0.2, 0],
    [0.8, -0.7, 1, -0.6, -0.3, 0.5],
    [-0.9, -0.5, -0.6, 1, 0.6, -0.7],
    [-0.8, 0.2, -0.3, 0.6, 1, -0.3],
    [0.4, 0, 0.5, -0.7, -0.3, 1]
])
n_assets = len(expected_returns)
base_corr = base_corr[:n_assets, :n_assets]
variances = stock_df["Variance"].to_numpy()
stds = np.sqrt(variances)
cov_matrix = np.outer(stds, stds) * base_corr
cov_matrix = (cov_matrix + cov_matrix.T) / 2
eigvals = np.linalg.eigvalsh(cov_matrix)
if np.any(eigvals < 0):
    cov_matrix += np.eye(n_assets) * (abs(min(eigvals)) + 1e-5)

# Sidebar controls
st.sidebar.header("Controls")
risk_aversion = st.sidebar.slider("Risk Aversion (λ)", 0.0, 5.0, 1.0, 0.1)
portfolio_count = st.sidebar.slider("Portfolios to Evaluate", 10, 200, 100, 10)

# Efficient frontier by sweeping lambda-weighted utility
solutions, utils, risks, returns = [], [], [], []
target_returns = np.linspace(min(expected_returns), max(expected_returns), portfolio_count)

x = cp.Variable(n_assets)
constraints = [cp.sum(x) == 1, x >= 0]
for alpha in target_returns:
    objective = cp.Maximize(expected_returns @ x - risk_aversion * cp.quad_form(x, cov_matrix))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if x.value is not None:
            solutions.append(x.value)
            risks.append(np.sqrt(x.value.T @ cov_matrix @ x.value))
            returns.append(expected_returns @ x.value)
            utils.append(expected_returns @ x.value - risk_aversion * x.value.T @ cov_matrix @ x.value)
    except:
        continue

# DataFrame
df = pd.DataFrame(solutions, columns=asset_names)
df["Expected Return"] = returns
df["Risk (Std Dev)"] = risks
df["Utility"] = utils

# Plot
st.subheader("📈 Efficient Frontier")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df["Risk (Std Dev)"], df["Expected Return"], color="blue", label="Efficient Frontier")
ax.set_xlabel("Risk (Std Dev)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier with Risk Aversion")
ax.grid(True)
ax.legend()
st.pyplot(fig)

if st.checkbox("Show Interactive Plot"):
    st.plotly_chart(px.scatter(df, x="Risk (Std Dev)", y="Expected Return", color="Utility", hover_data=asset_names), use_container_width=True)

# Glossary
st.subheader("📘 Glossary of Terms")
st.markdown("""
- **Expected Return**: Weighted average future return based on estimates.
- **Risk (Std Dev)**: The portfolio’s volatility, a measure of uncertainty.
- **Efficient Frontier**: Optimal portfolios that offer the highest return for each risk level.
- **Risk Aversion (λ)**: Degree to which an investor sacrifices return to avoid risk.
- **Utility**: A combined score reflecting return adjusted for risk: `U = Return − λ × Risk²`.
- **Covariance Matrix**: Shows how returns of assets move together.
- **Portfolio Weights**: Percent of capital allocated to each asset.
- **Constraint**: Mathematical limits imposed during optimization (e.g., no shorting).
""")