import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="FIN 330 Group Project", page_icon="📈", layout="wide")
st.title("📈 FIN 330 — Stock Analytics & Portfolio Dashboard")

# ============================================================
#  SIDEBAR — inputs for both parts
# ============================================================
with st.sidebar:
    st.header("Part 1 — Stock")
    ticker = st.text_input("Ticker", value="LMT").upper().strip()

    st.markdown("---")
    st.header("Part 2 — Portfolio")
    st.caption("Enter any 5 tickers. Weights must sum to 1.0")

    col_t, col_w = st.columns([1, 1])
    col_t.markdown("**Ticker**")
    col_w.markdown("**Weight**")

    t1 = col_t.text_input("t1", value="AAPL",  label_visibility="collapsed").upper().strip()
    w1 = col_w.number_input("w1", min_value=0.0, max_value=1.0, value=0.20, step=0.05, label_visibility="collapsed")

    t2 = col_t.text_input("t2", value="MSFT",  label_visibility="collapsed").upper().strip()
    w2 = col_w.number_input("w2", min_value=0.0, max_value=1.0, value=0.20, step=0.05, label_visibility="collapsed")

    t3 = col_t.text_input("t3", value="NVDA",  label_visibility="collapsed").upper().strip()
    w3 = col_w.number_input("w3", min_value=0.0, max_value=1.0, value=0.20, step=0.05, label_visibility="collapsed")

    t4 = col_t.text_input("t4", value="GOOGL", label_visibility="collapsed").upper().strip()
    w4 = col_w.number_input("w4", min_value=0.0, max_value=1.0, value=0.20, step=0.05, label_visibility="collapsed")

    t5 = col_t.text_input("t5", value="AMZN",  label_visibility="collapsed").upper().strip()
    w5 = col_w.number_input("w5", min_value=0.0, max_value=1.0, value=0.20, step=0.05, label_visibility="collapsed")

    total_w = round(w1 + w2 + w3 + w4 + w5, 2)
    if total_w != 1.0:
        st.warning(f"Weights sum to {total_w} — must equal 1.0")

    run = st.button("Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Set your inputs in the sidebar and click **Run Analysis**.")
    st.stop()

# ============================================================
#  PART 1 — INDIVIDUAL STOCK ANALYSIS
# ============================================================
st.header("Part 1 — Individual Stock Analysis")
st.subheader(f"Ticker: {ticker}")

# Step 1: Data Collection
with st.spinner(f"Downloading 6 months of data for {ticker}..."):
    try:
        data = yf.download(ticker, period="6mo", auto_adjust=False, progress=False)
        if data.empty:
            st.error(f"No data found for **{ticker}**.")
            st.stop()
        close = data["Close"].squeeze().dropna()
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

# Step 2: Trend Analysis
data["MA5"]  = data["Close"].squeeze().rolling(5).mean()
data["MA20"] = data["Close"].squeeze().rolling(20).mean()
data["MA50"] = data["Close"].squeeze().rolling(50).mean()

current_price = float(close.iloc[-1])
ma_20 = float(data["MA20"].squeeze().iloc[-1])
ma_50 = float(data["MA50"].squeeze().iloc[-1])

if current_price > ma_20 and current_price > ma_50:
    trend = "Strong Upward Trend"
elif current_price < ma_20 and current_price < ma_50:
    trend = "Strong Downward Trend"
else:
    trend = "Mixed Trend"

# Step 3: RSI
def compute_rsi(prices, window=14):
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

rsi_series  = compute_rsi(close)
current_rsi = float(rsi_series.iloc[-1])

if current_rsi > 70:
    rsi_signal, rsi_note, signal_color = "Overbought", "Possible Sell Signal", "red"
elif current_rsi < 30:
    rsi_signal, rsi_note, signal_color = "Oversold",   "Possible Buy Signal",  "green"
else:
    rsi_signal, rsi_note, signal_color = "Neutral",    "No clear signal",      "steelblue"

# Step 4: Volatility
daily_ret_p1 = close.pct_change().dropna()
vol_pct = float(daily_ret_p1.rolling(window=20).std().iloc[-1]) * np.sqrt(252) * 100

if vol_pct > 40:    vol_level = "High (> 40%)"
elif vol_pct >= 25: vol_level = "Medium (25%–40%)"
else:               vol_level = "Low (< 25%)"

# Step 5: Recommendation
buy_signals = sell_signals = 0
if trend == "Strong Upward Trend":   buy_signals  += 1
if trend == "Strong Downward Trend": sell_signals += 1
if current_rsi < 30: buy_signals  += 1
if current_rsi > 70: sell_signals += 1

if buy_signals > sell_signals:   recommendation = "🟢 BUY"
elif sell_signals > buy_signals: recommendation = "🔴 SELL"
else:                            recommendation = "🟡 HOLD"

reason = f"Trend: {trend} | RSI: {current_rsi:.1f} ({rsi_signal}) | Volatility: {vol_level}"

# Metric cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Current Price",  f"${current_price:.2f}")
c2.metric("20-Day MA",      f"${ma_20:.2f}")
c3.metric("50-Day MA",      f"${ma_50:.2f}")
c4.metric("RSI (14-day)",   f"{current_rsi:.1f}", rsi_signal)
c5.metric("Volatility",     f"{vol_pct:.1f}%",    vol_level)
c6.metric("Recommendation", recommendation)
st.caption(f"Trend: **{trend}** | {reason}")
st.markdown("---")

# Chart: Price + Moving Averages
st.subheader("Price & Moving Averages")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(close.index, close.to_numpy(), label="Close Price", color="steelblue", linewidth=1.8)
ax1.plot(data["MA5"].squeeze().index,  data["MA5"].squeeze().to_numpy(),  label="MA5",  color="green",  linewidth=1.2, linestyle="--")
ax1.plot(data["MA20"].squeeze().index, data["MA20"].squeeze().to_numpy(), label="MA20", color="orange", linewidth=1.2, linestyle="--")
ax1.plot(data["MA50"].squeeze().index, data["MA50"].squeeze().to_numpy(), label="MA50", color="red",    linewidth=1.2, linestyle="--")
ax1.set_title(f"{ticker} — Moving Average vs Price | Trend: {trend}", fontweight="bold")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# Chart: RSI
st.subheader("RSI (14-Day)")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.axhspan(70, 100, color="red",   alpha=0.07)
ax2.axhspan(30, 70,  color="gray",  alpha=0.05)
ax2.axhspan(0,  30,  color="green", alpha=0.07)
ax2.axhline(70, color="red",   linestyle="--", linewidth=1.0, label="Overbought (70)")
ax2.axhline(30, color="green", linestyle="--", linewidth=1.0, label="Oversold (30)")
ax2.axhline(50, color="gray",  linestyle=":",  linewidth=0.7, label="Midline (50)")
ax2.plot(rsi_series.index, rsi_series.to_numpy(), color="purple", linewidth=1.4, label="RSI (14-day)")
ax2.scatter([rsi_series.index[-1]], [current_rsi], color=signal_color, zorder=5, s=60)
ax2.annotate(
    f"  RSI = {current_rsi:.1f}\n  {rsi_signal}\n  ({rsi_note})",
    xy=(rsi_series.index[-1], current_rsi),
    xytext=(-90, 20), textcoords="offset points",
    fontsize=9, color=signal_color, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=signal_color, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=signal_color, alpha=0.85)
)
xmax = rsi_series.index[-1]
ax2.text(xmax, 85, "OVERBOUGHT", color="red",     fontsize=8, fontweight="bold", va="center", ha="right", alpha=0.7)
ax2.text(xmax, 50, "NEUTRAL",    color="dimgray", fontsize=8, fontweight="bold", va="center", ha="right", alpha=0.7)
ax2.text(xmax, 15, "OVERSOLD",   color="green",   fontsize=8, fontweight="bold", va="center", ha="right", alpha=0.7)
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.25)
plt.tight_layout()
st.pyplot(fig2)

# ============================================================
#  PART 2 — PORTFOLIO PERFORMANCE DASHBOARD
# ============================================================
st.markdown("---")
st.header("Part 2 — Portfolio Performance Dashboard")

portfolio = {t1: w1, t2: w2, t3: w3, t4: w4, t5: w5}

end_date   = pd.Timestamp.now().normalize()
start_date = end_date - pd.DateOffset(years=1)

with st.spinner("Downloading 1 year of portfolio & benchmark data..."):
    prices = {}
    for symbol in portfolio:
        d = yf.download(symbol, start=start_date, end=end_date,
                        progress=False, auto_adjust=False, multi_level_index=False)
        prices[symbol] = d["Adj Close"]
    prices = pd.DataFrame(prices)

    benchmark = yf.download("^GSPC", start=start_date, end=end_date,
                             progress=False, auto_adjust=False, multi_level_index=False)
    benchmark = benchmark["Adj Close"]

# Return Calculations
daily_returns     = prices.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()
common_index      = daily_returns.index.intersection(benchmark_returns.index)
daily_returns     = daily_returns.loc[common_index]
benchmark_returns = benchmark_returns.loc[common_index]

portfolio_returns = pd.Series(0.0, index=daily_returns.index)
for symbol, weight in portfolio.items():
    portfolio_returns += daily_returns[symbol] * weight

# Performance Metrics
portfolio_total  = float((1 + portfolio_returns).prod() - 1)
benchmark_total  = float((1 + benchmark_returns).prod() - 1)
overall_vol      = float(portfolio_returns.std() * np.sqrt(252))
benchmark_vol    = float(benchmark_returns.std() * np.sqrt(252))
risk_free_rate   = 0.03
portfolio_sharpe = (portfolio_total - risk_free_rate) / overall_vol
benchmark_sharpe = (benchmark_total - risk_free_rate) / benchmark_vol
outperf          = portfolio_total - benchmark_total

# Metric cards
m1, m2, m3 = st.columns(3)
m1.metric("Portfolio Return",    f"{portfolio_total:.2%}", f"{outperf:+.2%} vs S&P 500")
m2.metric("Benchmark (S&P 500)", f"{benchmark_total:.2%}")
m3.metric("Outperformance",      f"{outperf:+.2%}")

m4, m5, m6 = st.columns(3)
m4.metric("Portfolio Volatility", f"{overall_vol:.2%}")
m5.metric("Benchmark Volatility", f"{benchmark_vol:.2%}")
m6.metric("Portfolio Sharpe",     f"{portfolio_sharpe:.2f}", f"vs Benchmark {benchmark_sharpe:.2f}")

st.markdown("---")

# Chart: Portfolio vs Benchmark
st.subheader("Cumulative Growth — Portfolio vs Benchmark")
cum_port  = (1 + portfolio_returns).cumprod()
cum_bench = (1 + benchmark_returns).cumprod()

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(cum_port.index,  cum_port.to_numpy(),  label="Portfolio",        color="steelblue",  linewidth=2)
ax3.plot(cum_bench.index, cum_bench.to_numpy(), label="S&P 500 Benchmark", color="darkorange", linewidth=1.5, linestyle="--")
ax3.set_title("Portfolio vs Benchmark — Cumulative Growth of $1", fontweight="bold")
ax3.set_ylabel("Growth ($)")
ax3.set_xlabel("Date")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# Step 6: Interpretation
st.subheader("Interpretation")

# --- Returns comparison ---
r1, r2, r3 = st.columns(3)
r1.metric("Portfolio Return",    f"{portfolio_total:.2%}")
r2.metric("Benchmark Return",    f"{benchmark_total:.2%}")
r3.metric("Outperformance",      f"{outperf:+.2%}",
          "Above benchmark" if outperf >= 0 else "Below benchmark")

if outperf > 0:
    st.success(f"The portfolio **outperformed** the S&P 500 by **{outperf:.2%}** "
               f"({portfolio_total:.2%} portfolio vs {benchmark_total:.2%} benchmark).")
else:
    st.error(f"The portfolio **underperformed** the S&P 500 by **{abs(outperf):.2%}** "
             f"({portfolio_total:.2%} portfolio vs {benchmark_total:.2%} benchmark).")

st.markdown("---")

# --- Volatility comparison ---
v1, v2, v3 = st.columns(3)
v1.metric("Portfolio Volatility",  f"{overall_vol:.2%}")
v2.metric("Benchmark Volatility",  f"{benchmark_vol:.2%}")
vol_diff = overall_vol - benchmark_vol
v3.metric("Volatility Difference", f"{vol_diff:+.2%}",
          "More risky" if vol_diff > 0 else "Less risky")

if overall_vol > benchmark_vol:
    st.warning(f"The portfolio was **more volatile** (riskier) than the S&P 500 — "
               f"**{overall_vol:.2%}** vs **{benchmark_vol:.2%}** "
               f"({vol_diff:+.2%} difference).")
else:
    st.info(f"The portfolio was **less volatile** (less risky) than the S&P 500 — "
            f"**{overall_vol:.2%}** vs **{benchmark_vol:.2%}** "
            f"({vol_diff:+.2%} difference).")

st.markdown("---")

# --- Sharpe ratio comparison ---
s1, s2, s3 = st.columns(3)
s1.metric("Portfolio Sharpe Ratio",  f"{portfolio_sharpe:.2f}")
s2.metric("Benchmark Sharpe Ratio",  f"{benchmark_sharpe:.2f}")
sharpe_diff = portfolio_sharpe - benchmark_sharpe
s3.metric("Sharpe Difference", f"{sharpe_diff:+.2f}",
          "More efficient" if sharpe_diff > 0 else "Less efficient")

if portfolio_sharpe > benchmark_sharpe:
    st.success(f"The portfolio was **more efficient** than the benchmark — "
               f"Sharpe ratio **{portfolio_sharpe:.2f}** vs **{benchmark_sharpe:.2f}**, "
               f"meaning it generated better return per unit of risk.")
elif portfolio_sharpe > 0:
    st.warning(f"The portfolio had a positive Sharpe ratio (**{portfolio_sharpe:.2f}**) but was "
               f"**less efficient** than the benchmark (**{benchmark_sharpe:.2f}**) — "
               f"the benchmark generated more return per unit of risk.")
else:
    st.error(f"The portfolio had a **negative Sharpe ratio** (**{portfolio_sharpe:.2f}**) vs "
             f"benchmark (**{benchmark_sharpe:.2f}**) — returns did not compensate for the risk taken.")

