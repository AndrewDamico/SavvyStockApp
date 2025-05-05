import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cvxpy as cp

st.set_page_config(page_title="Savvy Stock Portfolio Optimizer – Multi-Risk", layout="wide")
st.title("📈 Savvy Stock Portfolio Optimizer – Multi-Risk Models")

# --- Sidebar: Data Profile ---
st.sidebar.header("Data Profile")
profile = st.sidebar.radio(
    "Choose Stock Data Profile",
    ["Original (Case-Based)", "Exploratory (Broader Spread)", "Use My Own Data"]
)
if profile == "Original (Case-Based)":
    stock_data = {
        "Stock": ["BB","LOP","ILI","HEAL","QUI","AUA"],
        "Start Price": [60,127,4,50,150,20],
        "Expected Price": [72,180.34,8,75,219,26],
        "Variance": [0.032,0.1,0.333,0.125,0.065,0.08],
    }
    editable=False; add_rows=False
elif profile == "Exploratory (Broader Spread)":
    stock_data = {
        "Stock": ["X","Y","Z","Alpha","Beta","Gamma"],
        "Start Price": [50,100,30,80,45,25],
        "Expected Price": [65,115,35,110,60,30],
        "Variance": [0.04,0.08,0.09,0.06,0.07,0.05],
    }
    editable=False; add_rows=False
else:
    tickers = [t.strip().upper() for t in st.sidebar.text_input(
        "Enter tickers, comma-separated", "AAPL, MSFT, GOOG"
    ).split(",") if t.strip()]
    N = max(1,len(tickers))
    stock_data = {
        "Stock": tickers or ["AAPL"],
        "Start Price": [100.0]*N,
        "Expected Price": [110.0]*N,
        "Variance": [0.05]*N
    }
    editable=True; add_rows=True

# --- Sidebar: Optimization & Risk Settings ---
st.sidebar.header("Optimization & Risk Settings")
min_return = st.sidebar.slider("Min Expected Return", 0.01, 1.0, 0.2, 0.01)
risk_model = st.sidebar.selectbox("Select Risk Model for Optimization",
                                  ["Variance", "CVaR (approx)", "Max Drawdown (approx)"])
st.sidebar.markdown("""
**Risk Models Defined:**
- **Variance**: Portfolio volatility.  
- **CVaR**: Avg. loss beyond a worst-case percentile (requires simulation).  
- **Max Drawdown**: Largest peak-to-trough drop (via simulated paths).
""")
num_sims = st.sidebar.number_input("Simulation Paths", 100, 10000, 1000, 100)
horizon = st.sidebar.number_input("Simulation Horizon (days)", 30, 252, 126, 1)
alpha = st.sidebar.slider("CVaR Confidence Level", 0.90, 0.99, 0.95, 0.01)

# --- Prepare Data ---
df_input = pd.DataFrame(stock_data)
st.subheader("Stock Parameters")
stock_df = st.data_editor(df_input, num_rows="dynamic" if add_rows else "fixed", use_container_width=True)

# Validate and compute metrics
stock_df["Start Price"] = pd.to_numeric(stock_df["Start Price"], errors="coerce")
stock_df["Expected Price"] = pd.to_numeric(stock_df["Expected Price"], errors="coerce")
stock_df["Variance"] = pd.to_numeric(stock_df["Variance"], errors="coerce")
if stock_df.isnull().any(axis=1).any():
    st.error("Please ensure all price and variance fields are numeric.")
    st.stop()

names = stock_df["Stock"].tolist()
mu = ((stock_df["Expected Price"] - stock_df["Start Price"]) / stock_df["Start Price"]).to_numpy()
sigma2 = stock_df["Variance"].to_numpy()
Sigma = np.diag(sigma2)  # No correlation by default

# --- Mean-Variance Optimization (always computed) ---
N = len(names)
x = cp.Variable(N)
constraints = [cp.sum(x)==1, x>=0]
rets = np.linspace(mu.min(), mu.max(), 100)
frontier = []
for R in rets:
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints + [mu @ x >= R])
    prob.solve(solver=cp.ECOS)
    if x.value is not None:
        frontier.append((R, float(np.sqrt(x.value.T @ Sigma @ x.value))))
    else:
        frontier.append((R, None))

df_front = pd.DataFrame(frontier, columns=["Return","Risk"])
df_valid = df_front.dropna()

# --- Handle no feasible portfolios ---
feasible = df_valid[df_valid["Return"] >= min_return]
if feasible.empty:
    st.warning("⚠️ No feasible portfolio found for the selected minimum return.")
    st.stop()

# --- Select Portfolio at min_return ---
opt_row = feasible.iloc[0]
prob_opt = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints + [mu @ x >= opt_row["Return"]])
prob_opt.solve(solver=cp.ECOS)
w_opt = np.array(x.value).flatten()

# --- Simulation for CVaR/Drawdown if needed ---
term_rets = None; CVaR_val = None; avg_dd = None
if risk_model != "Variance":
    dt = 1/252
    sims = np.random.multivariate_normal(mu*dt, np.diag(sigma2*dt), (num_sims, horizon))
    port_vals = np.cumprod(1 + sims @ w_opt.reshape(-1,1), axis=1)
    term_rets = port_vals[:,-1] - 1
    losses = -term_rets
    drawdowns = np.max(1 - port_vals / np.maximum.accumulate(port_vals,axis=1), axis=1)
    if "CVaR" in risk_model:
        VaR = np.quantile(losses, alpha)
        CVaR_val = losses[losses >= VaR].mean()
    else:
        avg_dd = drawdowns.mean()

# --- Display Results ---
st.subheader(f"Optimal Portfolio for Min Return {min_return:.2f}")
st.table(pd.DataFrame({"Weight": w_opt}, index=names))

st.markdown("**Risk Metric:**")
if risk_model == "Variance":
    st.write(f"Standard Deviation: {opt_row['Risk']:.4f}")
elif "CVaR" in risk_model:
    st.write(f"CVaR @{int(alpha*100)}%: {CVaR_val:.4f}")
else:
    st.write(f"Avg Max Drawdown: {avg_dd:.4f}")

# --- Plots ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Efficient Frontier (Variance)")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_valid["Risk"], df_valid["Return"], label="Frontier")
    ax.scatter(opt_row["Risk"], opt_row["Return"], color="red", label="Selected")
    ax.set_xlabel("Risk (Std Dev)"); ax.set_ylabel("Return"); ax.legend()
    st.pyplot(fig)
with col2:
    if risk_model == "Variance":
        st.markdown("### Return Distribution (Sample)")
        st.write("Switch to CVaR or Drawdown for simulations.")
    elif "CVaR" in risk_model:
        st.markdown("### Loss Distribution for CVaR")
        st.plotly_chart(px.histogram(losses, nbins=50, title="Loss Distribution"), use_container_width=True)
    else:
        st.markdown("### Drawdown Distribution")
        st.plotly_chart(px.histogram(drawdowns, nbins=50, title="Drawdown Distribution"), use_container_width=True)

# --- Asset Weights by Return Target ---
st.subheader("Asset Weights by Return Target (Mean-Variance Frontier)")
fig_wt, ax_wt = plt.subplots(figsize=(6,4))
# Use the mean-variance weights DataFrame (df_mv) to plot allocations
# df_mv contains weights for each asset and columns [asset names, 'Return', 'StdDev']
df_weights = df_mv.dropna()
for name in names:
    ax_wt.plot(df_weights["Return"], df_weights[name], label=name)
ax_wt.set_xlabel("Expected Return"); ax_wt.set_ylabel("Weight")
ax_wt.set_title("Asset Allocation vs Return")
ax_wt.legend()
st.pyplot(fig_wt)

# --- Glossary ---
st.subheader("📘 Glossary of Terms")
st.markdown("""
- **Expected Return**: Predicted % gain.  
- **Std Dev (Variance)**: Total volatility measure.  
- **CVaR**: Average of worst losses beyond a percentile.  
- **Max Drawdown**: Largest peak-to-trough loss.  
- **Simulation Paths**: Monte Carlo return scenarios.  
- **Optimization**: Minimize selected risk metric under return constraint.  
""" )
