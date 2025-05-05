
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

df_input = pd.DataFrame(stock_data)
st.subheader("Stock Parameters")
stock_df = st.data_editor(df_input, num_rows="dynamic" if add_rows else "fixed", use_container_width=True)
stock_df["Start Price"] = pd.to_numeric(stock_df["Start Price"],errors="coerce")
stock_df["Expected Price"] = pd.to_numeric(stock_df["Expected Price"],errors="coerce")
stock_df["Variance"] = pd.to_numeric(stock_df["Variance"],errors="coerce")
if stock_df.isnull().any().any():
    st.error("Please fill all price & variance values.")
    st.stop()

names = stock_df["Stock"].tolist()
mu = ((stock_df["Expected Price"]-stock_df["Start Price"])/stock_df["Start Price"]).to_numpy()
sigma2 = stock_df["Variance"].to_numpy()
Sigma = np.diag(sigma2)

# --- Sidebar: Optimization & Risk Settings ---
st.sidebar.header("Optimization & Risk")
min_return = st.sidebar.slider("Min Expected Return",0.01,1.0,0.2,0.01)
risk_model = st.sidebar.selectbox("Risk Model",["Variance","CVaR","Max Drawdown"])
num_sims = st.sidebar.number_input("Simulation Paths",100,10000,1000,100)
horizon = st.sidebar.number_input("Simulation Horizon (days)",30,252,126,1)
alpha = st.sidebar.slider("CVaR Confidence Level",0.90,0.99,0.95,0.01)

# --- Mean-Variance Frontier ---
N = len(names)
x = cp.Variable(N)
cons = [cp.sum(x)==1, x>=0]
rets = np.linspace(min(mu),max(mu),100)
risks_mv,weights_mv = [],[]
for R in rets:
    prob = cp.Problem(cp.Minimize(cp.quad_form(x,Sigma)), cons+[mu@x>=R])
    prob.solve(solver=cp.ECOS)
    if x.value is not None:
        risks_mv.append(np.sqrt(float(x.value.T@Sigma@x.value)))
        weights_mv.append(x.value)
    else:
        risks_mv.append(None)
        weights_mv.append([None]*N)
df_mv = pd.DataFrame(weights_mv,columns=names)
df_mv["Return"]=rets; df_mv["StdDev"]=risks_mv
df_valid = df_mv.dropna()

# --- Select optimal portfolio row ---
opt = df_valid[df_valid["Return"]>=min_return].iloc[0]
w_opt = opt[names].to_numpy()
opt_r = opt["Return"]; opt_sd = opt["StdDev"]

# --- Simulation for CVaR & Drawdown ---
dt=1/252
# simulate returns
simrets = np.random.multivariate_normal(mu*dt,np.diag(sigma2*dt),(num_sims,horizon))
# portfolio value paths
pv = np.cumprod(1 + simrets @ w_opt.reshape(-1,1),axis=1)
term_ret = pv[:,-1]-1
# compute CVaR
losses = -term_ret
VaR = np.quantile(losses,alpha)
CVaR = losses[losses>=VaR].mean()
# compute max drawdown
drawdowns = np.max(1 - pv/np.maximum.accumulate(pv,axis=1),axis=1)
avg_dd = drawdowns.mean()

# --- Display ---
st.subheader(f"Optimal Portfolio (Min Return {min_return:.2f})")
st.write(pd.Series(w_opt,index=names).rename("Weight"))

st.markdown("**Risk Metric:**")
if risk_model=="Variance":
    st.write(f"Std Dev: {opt_sd:.4f}")
elif risk_model=="CVaR":
    st.write(f"CVaR @ {alpha*100:.0f}%: {CVaR:.4f}")
else:
    st.write(f"Avg Max Drawdown: {avg_dd:.4f}")

# --- Plots ---
col1,col2=st.columns(2)
with col1:
    st.markdown("### Efficient Frontier")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(df_valid["StdDev"],df_valid["Return"],label="Frontier")
    ax.scatter(opt_sd,opt_r,color="red",label="Selected")
    ax.set_xlabel("Std Dev"); ax.set_ylabel("Return"); ax.legend()
    st.pyplot(fig)
with col2:
    if risk_model=="Variance":
        st.markdown("### Return Distribution")
        st.plotly_chart(px.histogram(term_ret,title="Simulated Terminal Returns"),use_container_width=True)
    elif risk_model=="CVaR":
        st.markdown("### Loss Distribution")
        st.plotly_chart(px.histogram(losses,nbins=50,title="Losses for CVaR"),use_container_width=True)
    else:
        st.markdown("### Example Drawdown Path")
        fig2,ax2=plt.subplots(figsize=(6,4))
        ax2.plot(drawdowns[:horizon])
        ax2.set_title("Sample Drawdown"); ax2.set_ylabel("Drawdown")
        st.pyplot(fig2)

# --- Glossary ---
st.subheader("📘 Glossary")
st.markdown("""
- **Expected Return**: Predicted % gain based on prices.
- **Standard Deviation**: Total volatility (Variance model).
- **CVaR**: Avg loss beyond Value-at-Risk at confidence α.
- **Max Drawdown**: Largest peak-to-trough drop.
- **Simulation Paths**: Monte Carlo for returns estimation.
- **Optimization**: Minimize chosen risk metric under return constraint.
""")
