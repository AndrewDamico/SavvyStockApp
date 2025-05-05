import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cvxpy as cp
import yfinance as yf

st.set_page_config(page_title="Savvy Stock Portfolio Optimizer", layout="wide")
st.title("📈 Savvy Stock Portfolio Optimizer")

st.sidebar.header("Data Profile")
profile = st.sidebar.radio("Choose Stock Data Profile", ["Original (Case-Based)", "Exploratory (Broader Risk Spread)", "Use My Own Data"])

if profile == "Original (Case-Based)":
    stock_data = {
        "Stock": ["BB", "LOP", "ILI", "HEAL", "QUI", "AUA"],
        "Start Price": [60, 127, 4, 50, 150, 20],
        "Expected Price": [72, 180.34, 8, 75, 219, 26],
        "Variance": [0.032, 0.1, 0.333, 0.125, 0.065, 0.08],
    }
    editable = False
    add_rows = False
elif profile == "Exploratory (Broader Risk Spread)":
    stock_data = {
        "Stock": ["X", "Y", "Z", "Alpha", "Beta", "Gamma"],
        "Start Price": [50, 100, 30, 80, 45, 25],
        "Expected Price": [65, 115, 35, 110, 60, 30],
        "Variance": [0.04, 0.08, 0.09, 0.06, 0.07, 0.05],
    }
    editable = False
    add_rows = False
else:
    ticker_input = st.text_input("Enter stock tickers (comma-separated)", value="AAPL, MSFT, GOOG")
    tickers = [x.strip().upper() for x in ticker_input.split(",") if x.strip()]
    fetched_data = []

    if st.button("🔍 Fetch Real Stock Data"):
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).history(period="6mo")
                if not info.empty:
                    start_price = info["Close"][-2]
                    expected_price = info["Close"][-1]
                    variance = info["Close"].pct_change().var()
                    fetched_data.append((ticker, round(start_price, 2), round(expected_price, 2), round(variance, 4)))
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {e}")
        if fetched_data:
            stock_data = {
                "Stock": [t[0] for t in fetched_data],
                "Start Price": [t[1] for t in fetched_data],
                "Expected Price": [t[2] for t in fetched_data],
                "Variance": [t[3] for t in fetched_data],
            }
        else:
            stock_data = {"Stock": [], "Start Price": [], "Expected Price": [], "Variance": []}
    else:
        stock_data = {
            "Stock": ["AAPL", "MSFT", "GOOG"],
            "Start Price": [180, 320, 2900],
            "Expected Price": [200, 350, 3100],
            "Variance": [0.04, 0.06, 0.07],
        }

    editable = True
    add_rows = True

initial_data = pd.DataFrame(stock_data)
st.subheader("Stock Parameters")
stock_df = st.data_editor(initial_data, num_rows="dynamic" if add_rows else "fixed", use_container_width=True)
stock_df = stock_df.dropna()

stock_df["Start Price"] = pd.to_numeric(stock_df["Start Price"], errors="coerce")
stock_df["Expected Price"] = pd.to_numeric(stock_df["Expected Price"], errors="coerce")
stock_df["Variance"] = pd.to_numeric(stock_df["Variance"], errors="coerce")

expected_returns = ((stock_df["Expected Price"] - stock_df["Start Price"]) / stock_df["Start Price"]).to_numpy()
asset_names = stock_df["Stock"].tolist()

# ⛔️ Early exit if no valid data
if len(stock_df) == 0 or len(expected_returns) == 0:
    st.error("❌ No valid stock data available. Please add rows or check your inputs.")
    st.stop()

base_corr = np.identity(len(asset_names))
n_assets = len(expected_returns)
variances = stock_df["Variance"].to_numpy()
stds = np.sqrt(variances)
cov_matrix = np.outer(stds, stds) * base_corr
cov_matrix = (cov_matrix + cov_matrix.T) / 2
eigvals = np.linalg.eigvalsh(cov_matrix)
if np.any(eigvals < 0):
    cov_matrix += np.eye(n_assets) * (abs(min(eigvals)) + 1e-5)

# Sidebar Controls
st.sidebar.header("Optimization Controls")
min_return = st.sidebar.slider("Minimum Expected Return", 0.05, 0.60, 0.10, 0.01)
max_alloc_toggle = st.sidebar.checkbox("Enable Max Allocation Constraint?", value=True)
max_alloc_value = st.sidebar.slider("Max Allocation Per Stock (%)", 10, 100, 100, 5) / 100 if max_alloc_toggle else None
highlight_mvp = st.sidebar.checkbox("Highlight Minimum Variance Portfolio", value=True)
solver = st.sidebar.selectbox("Solver", options=["ECOS", "SCS", "OSQP"])
download_plotly = st.sidebar.checkbox("Enable Interactive Plotly Chart")

# Optimization
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

df = pd.DataFrame(solutions, columns=asset_names)
df["Expected Return"] = target_returns
df["Risk (Std Dev)"] = risks
df_valid = df.dropna()

# Optimal Portfolio
if not df_valid.empty and not df_valid[df_valid["Expected Return"] >= min_return].empty:
    selected_row = df_valid[df_valid["Expected Return"] >= min_return].iloc[0]
    st.subheader("📌 Optimal Portfolio at Minimum Return Target")
    st.write("Expected Return:", round(selected_row["Expected Return"], 3))
    st.write("Risk (Std Dev):", round(selected_row["Risk (Std Dev)"], 3))
    st.dataframe(selected_row[asset_names].T.rename("Weight (%)") * 100)
else:
    st.warning("⚠️ No feasible portfolio found. Try reducing the target return or increasing max allocation.")

# Efficient Frontier
st.subheader("📊 Efficient Frontier")
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(df["Risk (Std Dev)"], df["Expected Return"], label="Efficient Frontier", color="blue")
if highlight_mvp and not df_valid.empty:
    min_risk_idx = df_valid["Risk (Std Dev)"].idxmin()
    min_point = df_valid.loc[min_risk_idx]
    ax2.scatter(min_point["Risk (Std Dev)"], min_point["Expected Return"], color='red', label="Min Variance Portfolio", zorder=5)
ax2.set_xlabel("Risk (Std Dev)")
ax2.set_ylabel("Expected Return")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Weights Chart
st.subheader("📊 Asset Weights by Return Target")
fig_weights, ax_weights = plt.subplots(figsize=(8, 6))
for col in asset_names:
    ax_weights.plot(df["Expected Return"], df[col], label=col)
ax_weights.set_xlabel("Expected Return")
ax_weights.set_ylabel("Portfolio Weight")
ax_weights.set_title("Allocation Weights Across Efficient Frontier")
ax_weights.grid(True)
ax_weights.legend()
st.pyplot(fig_weights)

# Glossary
st.subheader("📘 Glossary of Terms")
st.markdown("""
- **Expected Return**: Projected gain based on stock price targets.
- **Risk (Standard Deviation)**: The volatility of the portfolio's return.
- **Efficient Frontier**: A curve showing optimal portfolios for each risk level.
- **Minimum Variance Portfolio**: The portfolio with the lowest risk.
- **Solver**: Mathematical engine used to optimize weights.
- **Max Allocation Constraint**: Limits how much capital can go into one stock.
- **Covariance Matrix**: Encodes how returns of assets move together.
- **Portfolio Weights**: Proportional investment in each asset.
""")