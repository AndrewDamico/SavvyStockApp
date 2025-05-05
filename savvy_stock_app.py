import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Savvy Stock Portfolio Optimizer", layout="wide")
st.title("📈 Savvy Stock Selection: Portfolio Optimizer")

# Define expected returns
expected_returns = np.array([
    (72 - 60) / 60,
    (127 * 1.42 - 127) / 127,
    (8 - 4) / 4,
    (75 - 50) / 50,
    (150 * 1.46 - 150) / 150,
    (26 - 20) / 20
], dtype=np.float64)
asset_names = ['BB', 'LOP', 'ILI', 'HEAL', 'QUI', 'AUA']

# Covariance matrix
raw_cov = np.array([
    [0.032, 0.005, 0.03, -0.031, -0.027, 0.01],
    [0.005, 0.1, -0.07, -0.05, 0.02, 0],
    [0.03, -0.07, 0.333, -0.11, -0.02, 0.042],
    [-0.031, -0.05, -0.11, 0.125, 0.05, -0.06],
    [-0.027, 0.02, -0.02, 0.05, 0.065, -0.02],
    [0.01, 0, 0.042, -0.06, -0.02, 0.08]
], dtype=np.float64)
cov_matrix = (raw_cov + raw_cov.T) / 2
eigvals = np.linalg.eigvalsh(cov_matrix)
if np.any(eigvals < 0):
    cov_matrix += np.eye(6) * (abs(min(eigvals)) + 1e-5)

# Sidebar controls
st.sidebar.header("Controls")
min_return = st.sidebar.slider("Minimum Expected Return", 0.10, 0.60, 0.30, 0.01)
max_alloc_toggle = st.sidebar.checkbox("Enable Max Allocation Constraint?", value=True)
max_alloc_value = st.sidebar.slider("Max Allocation Per Stock (%)", 10, 100, 100, 5) / 100 if max_alloc_toggle else None
highlight_mvp = st.sidebar.checkbox("Highlight Minimum Variance Portfolio", value=True)

# Efficient frontier setup
target_returns = np.linspace(0.10, 0.60, 200)
risks, solutions = [], []

x = cp.Variable((6,), name="weights")
constraints = [cp.sum(x) == 1, x >= 0]
if max_alloc_toggle:
    constraints.append(x <= max_alloc_value)

for target in target_returns:
    prob_constraints = constraints + [expected_returns @ x >= target]
    objective = cp.Minimize(cp.quad_form(x, cov_matrix))
    problem = cp.Problem(objective, prob_constraints)
    try:
        problem.solve()
        if x.value is not None:
            risks.append(np.sqrt(float(x.value.T @ cov_matrix @ x.value)))
            solutions.append(x.value)
        else:
            risks.append(None)
            solutions.append([None]*6)
    except Exception:
        risks.append(None)
        solutions.append([None]*6)

# DataFrame output
df = pd.DataFrame(solutions, columns=asset_names)
df["Expected Return"] = target_returns
df["Risk (Std Dev)"] = risks
df_valid = df.dropna()

# Check for valid solutions
if not df_valid.empty and not df_valid[df_valid["Expected Return"] >= min_return].empty:
    selected_row = df_valid[df_valid["Expected Return"] >= min_return].iloc[0]
    st.subheader("Optimal Portfolio at Selected Minimum Return")
    st.write("📌 Expected Return:", round(selected_row["Expected Return"], 3))
    st.write("📉 Risk (Std Dev):", round(selected_row["Risk (Std Dev)"], 3))
    st.dataframe(selected_row[asset_names].T.rename("Weight (%)") * 100)
else:
    st.warning("⚠️ No feasible portfolio found for this expected return and constraint combination. Try reducing the target return or increasing max allocation.")

# Side-by-side plots
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