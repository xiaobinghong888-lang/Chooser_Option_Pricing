import numpy as np
import pandas as pd
import streamlit as st
import os

from scripts.data_loader import load_latest_pricing_inputs

st.set_page_config(page_title="Chooser Option Pricing Tool", layout="wide")


# ----------------------------
# Pricing functions
# ----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


def bs_call_price(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)


def chooser_option_price(S, K, r, q, sigma, T, tc):
    if tc <= 0 or tc >= T:
        return np.nan
    adjusted_strike = K * np.exp(-(r - q) * (T - tc))
    call_part = bs_call_price(S, K, r, q, sigma, T)
    put_part = bs_put_price(S, adjusted_strike, r, q, sigma, tc)
    return call_part + put_part


# ----------------------------
# Load latest default data
# ----------------------------
latest_info = load_latest_pricing_inputs()

default_date = latest_info["Date"]
default_S = latest_info["JPM_Close"]
default_r = latest_info["RiskFree_1Y"]
default_sigma_base = latest_info["BaselineSigma_Annual"]
default_sigma_ml = latest_info["FinalPredictedSigma_Annual"]


# ----------------------------
# Optional file loaders
# ----------------------------
def safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


pricing_history_df = safe_read_csv("data/processed/final_ml_enhanced_bsm_prices.csv")
benchmark_summary_df = safe_read_csv("outputs/final_benchmark_error_summary.csv")


# ----------------------------
# Header
# ----------------------------
st.title("Chooser Option Pricing Tool")
st.markdown("Week 8 finalization: baseline BSM vs ML-enhanced BSM chooser pricing dashboard")

# ----------------------------
# Latest loaded data
# ----------------------------
st.markdown("### Latest Loaded Data")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Latest Date", default_date)

with c2:
    st.metric("Latest JPM_Close", f"{default_S:.4f}")

with c3:
    st.metric("Latest RiskFree_1Y", f"{default_r:.4f}")

with c4:
    st.metric("Latest Baseline Sigma", f"{default_sigma_base:.4f}")

with c5:
    st.metric("Latest ML Sigma", f"{default_sigma_ml:.4f}")

st.markdown("---")

# ----------------------------
# Inputs
# ----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Contract Inputs")
    S = st.number_input("Underlying price S", min_value=0.01, value=float(default_S), step=1.0)
    K = st.number_input("Strike K", min_value=0.01, value=150.0, step=1.0)
    T = st.number_input("Maturity T (years)", min_value=0.01, value=1.0, step=0.1)
    tc = st.number_input("Chooser decision time tc", min_value=0.001, value=0.5, step=0.1)

with right:
    st.subheader("Market Inputs")
    r = st.number_input("Risk-free rate r", value=float(default_r), step=0.005, format="%.4f")
    q = st.number_input("Dividend yield q", value=0.02, step=0.005, format="%.4f")
    sigma_base = st.number_input(
        "Baseline sigma",
        min_value=0.0001,
        value=float(default_sigma_base),
        step=0.01,
        format="%.4f",
    )
    sigma_ml = st.number_input(
        "ML-enhanced sigma",
        min_value=0.0001,
        value=float(default_sigma_ml),
        step=0.01,
        format="%.4f",
    )

# ----------------------------
# Core pricing outputs
# ----------------------------
baseline_price = chooser_option_price(S, K, r, q, sigma_base, T, tc)
ml_price = chooser_option_price(S, K, r, q, sigma_ml, T, tc)

price_gap = ml_price - baseline_price
abs_price_gap = abs(price_gap)
rel_price_gap = price_gap / baseline_price if pd.notna(baseline_price) and baseline_price != 0 else np.nan

sigma_gap = sigma_ml - sigma_base
abs_sigma_gap = abs(sigma_gap)

st.markdown("---")
st.subheader("Pricing Outputs")

r1, r2, r3 = st.columns(3)
with r1:
    st.metric("Baseline BSM Chooser Price", f"{baseline_price:.4f}" if pd.notna(baseline_price) else "NaN")
with r2:
    st.metric("ML-Enhanced Chooser Price", f"{ml_price:.4f}" if pd.notna(ml_price) else "NaN")
with r3:
    st.metric("ML - Baseline", f"{price_gap:.4f}" if pd.notna(price_gap) else "NaN")

# ----------------------------
# Error margin / adjustment block
# ----------------------------
st.markdown("---")
st.subheader("Error Margin / Adjustment Summary")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Absolute Price Gap", f"{abs_price_gap:.6f}" if pd.notna(abs_price_gap) else "NaN")
with m2:
    st.metric("Relative Price Gap (%)", f"{100 * rel_price_gap:.4f}%" if pd.notna(rel_price_gap) else "NaN")
with m3:
    st.metric("Sigma Gap", f"{sigma_gap:.6f}" if pd.notna(sigma_gap) else "NaN")
with m4:
    st.metric("Absolute Sigma Gap", f"{abs_sigma_gap:.6f}" if pd.notna(abs_sigma_gap) else "NaN")

if pd.notna(rel_price_gap):
    if abs(rel_price_gap) < 0.001:
        st.info("Current ML pricing adjustment is economically very small relative to the baseline BSM price.")
    elif abs(rel_price_gap) < 0.01:
        st.info("Current ML pricing adjustment is modest relative to the baseline BSM price.")
    else:
        st.info("Current ML pricing adjustment is relatively large under the chosen parameter combination.")

# ----------------------------
# Scenario quick check
# ----------------------------
st.markdown("---")
st.subheader("Scenario Quick Check")

vol_up_base = chooser_option_price(S, K, r, q, sigma_base * 1.5, T, tc)
rate_up_base = chooser_option_price(S, K, r + 0.02, q, sigma_base, T, tc)
combined_base = chooser_option_price(S, K, r + 0.02, q, sigma_base * 1.5, T, tc)

vol_up_ml = chooser_option_price(S, K, r, q, sigma_ml * 1.5, T, tc)
rate_up_ml = chooser_option_price(S, K, r + 0.02, q, sigma_ml, T, tc)
combined_ml = chooser_option_price(S, K, r + 0.02, q, sigma_ml * 1.5, T, tc)

scenario_df = pd.DataFrame({
    "Scenario": ["Base", "Vol Spike", "Rate Hike", "Combined Shock"],
    "Baseline Price": [baseline_price, vol_up_base, rate_up_base, combined_base],
    "ML Price": [ml_price, vol_up_ml, rate_up_ml, combined_ml],
})
scenario_df["ML - Baseline"] = scenario_df["ML Price"] - scenario_df["Baseline Price"]
scenario_df["Relative Gap (%)"] = 100 * scenario_df["ML - Baseline"] / scenario_df["Baseline Price"]

st.dataframe(scenario_df, use_container_width=True)

# ----------------------------
# Dashboard charts
# ----------------------------
st.markdown("---")
st.subheader("Visualization Dashboard")

tab1, tab2, tab3 = st.tabs(["Price Trend", "Sigma Sensitivity", "Performance Metrics"])

with tab1:
    st.markdown("#### Historical Price Trend: Baseline vs ML")
    if pricing_history_df is not None and {"Date", "ChooserPrice_BSM_Baseline", "ChooserPrice_GBR_Final"}.issubset(pricing_history_df.columns):
        price_plot_df = pricing_history_df[["Date", "ChooserPrice_BSM_Baseline", "ChooserPrice_GBR_Final"]].copy()
        price_plot_df["Date"] = pd.to_datetime(price_plot_df["Date"])
        price_plot_df = price_plot_df.sort_values("Date")
        price_plot_df = price_plot_df.rename(columns={
            "ChooserPrice_BSM_Baseline": "Baseline BSM",
            "ChooserPrice_GBR_Final": "ML-Enhanced BSM"
        })
        st.line_chart(price_plot_df.set_index("Date"))
    else:
        st.warning("Historical pricing file not available or required columns are missing.")

with tab2:
    st.markdown("#### Baseline Chooser Price Sensitivity to Sigma")
    sigma_grid = np.linspace(0.05, 0.60, 50)
    baseline_curve = [chooser_option_price(S, K, r, q, s, T, tc) for s in sigma_grid]
    ml_curve = [chooser_option_price(S, K, r, q, s, T, tc) for s in sigma_grid]

    sigma_chart_df = pd.DataFrame({
        "Sigma": sigma_grid,
        "Baseline Price": baseline_curve,
        "ML-style Price Curve": ml_curve
    })
    st.line_chart(sigma_chart_df.set_index("Sigma"))

with tab3:
    st.markdown("#### Performance Metrics")
    if benchmark_summary_df is not None:
        st.dataframe(benchmark_summary_df, use_container_width=True)
    else:
        perf_df = pd.DataFrame({
            "Metric": [
                "Current Baseline Price",
                "Current ML Price",
                "Absolute Price Gap",
                "Relative Price Gap (%)",
                "Absolute Sigma Gap"
            ],
            "Value": [
                baseline_price,
                ml_price,
                abs_price_gap,
                100 * rel_price_gap if pd.notna(rel_price_gap) else np.nan,
                abs_sigma_gap
            ]
        })
        st.dataframe(perf_df, use_container_width=True)
        st.caption("Fallback performance panel shown because benchmark summary file was not found.")