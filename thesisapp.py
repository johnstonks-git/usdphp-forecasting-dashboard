import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PRED_DIR = BASE_DIR / "predictions"

st.set_page_config(
    page_title="USD-PHP Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

# ═════════════════════════════════════════════════════════════════════════════
# THESIS DATA  (Tables 7, 9, 12 and section 4.1.1 text)
# ═════════════════════════════════════════════════════════════════════════════
HORIZONS = [1, 3, 7, 14, 21, 30]
H_LABELS = ["1-Day", "3-Day", "7-Day", "14-Day", "21-Day", "30-Day"]
MODELS   = ["N-BEATSx", "TFT", "Constrained Ensemble", "Naive Baseline"]

METRICS = {
    "MAE": {
        "N-BEATSx":            [0.1777, 0.2679, 0.3542, 0.5192, 0.6478, 0.8459],
        "TFT":                 [0.1681, 0.1651, 0.1752, 1.3390, 1.2313, 1.0709],
        "Constrained Ensemble":[0.1422, 0.1562, 0.1634, 0.3715, 0.5590, 0.7965],
        "Naive Baseline":      [0.1565, 0.2546, 0.3834, 0.5265, 0.7119, 0.9463],
    },
    "MSE": {
        "N-BEATSx":            [0.0517, 0.1216, 0.2184, 0.4687, 0.7082, 1.1510],
        "TFT":                 [0.0443, 0.0397, 0.0452, 2.5037, 2.2974, 1.7512],
        "Constrained Ensemble":[0.0326, 0.0352, 0.0380, 0.2205, 0.4477, 0.9200],
        "Naive Baseline":      [0.0431, 0.1156, 0.2671, 0.4688, 0.7471, 1.1507],
    },
    "MAPE": {
        "N-BEATSx":            [0.0031, 0.0047, 0.0062, 0.0091, 0.0113, 0.0147],
        "TFT":                 [0.0029, 0.0029, 0.0030, 0.0234, 0.0213, 0.0186],
        "Constrained Ensemble":[0.0025, 0.0026, 0.0027, 0.0065, 0.0097, 0.0139],
        "Naive Baseline":      [0.0027, 0.0044, 0.0067, 0.0092, 0.0125, 0.0165],
    },
    "R²": {
        "N-BEATSx":            [ 0.9532,  0.8899,  0.8019,  0.5739,  0.3507, -0.0675],
        "TFT":                 [ 0.9597,  0.9639,  0.9586, -1.3225, -1.2355, -0.7614],
        "Constrained Ensemble":[ 0.9694,  0.9676,  0.9660,  0.7953,  0.5643, -0.0643],
        "Naive Baseline":      [ 0.9610,  0.8956,  0.7610,  0.5945,  0.3505, -0.0558],
    },
}

ENSEMBLE_WEIGHTS = {
    1: (0.4013, 0.5987), 3: (0.1880, 0.8120), 7:  (0.1380, 0.8620),
    14:(0.9577, 0.0423), 21:(1.0000, 0.0000), 30: (0.9382, 0.0618),
}

NBEATSX_LOFO = {
    1:  [("Core Inflation", 21.20), ("Gasoline Prices", 20.29),
         ("Overnight Deposit Rate", 18.89), ("Inflation Rate", 12.40), ("Imports", 5.13)],
    3:  [("Interest Rate", 40.40), ("Inflation Rate", 24.94),
         ("Gasoline Prices", 17.46), ("Overnight Deposit Rate", 17.20)],
    7:  [("Inflation Rate", 42.56), ("Overnight Deposit Rate", 26.44),
         ("Core Inflation", 22.80), ("Balance of Trade", 18.63), ("Price Group", 9.76)],
    14: [("Inflation Rate", 33.62), ("Gasoline Prices", 23.01),
         ("Overnight Deposit Rate", 17.10), ("Price Group", 8.81), ("Balance of Trade", 6.98)],
    21: [("Gasoline Prices", 33.62), ("Core Inflation", 23.01),
         ("Cash Remittances", 17.10), ("Overnight Deposit Rate", 8.81), ("Interest Rate", 6.98)],
    30: [("Balance of Trade", 20.15), ("Cash Remittances", 16.62),
         ("Inflation Rate", 15.10), ("Exports", 9.81), ("Gasoline Prices", 8.70)],
}

TFT_ENCODER = {
    1:  [("Inflation Rate", 34.50), ("Gasoline Prices", 29.91),
         ("Close", 5.26), ("Exports", 4.90), ("High", 3.80)],
    3:  [("Interest Rate", 95.20), ("Rel. Time Index", 0.82),
         ("Core Inflation", 0.55), ("Cash Remittances", 0.37), ("Forex Reserves", 0.33)],
    7:  [("High", 70.00), ("Interest Rate", 12.07),
         ("Low", 2.42), ("Rel. Time Index", 2.19), ("Open", 1.91)],
    14: [("Interest Rate", 97.82), ("Core Inflation", 0.30),
         ("High", 0.18), ("Overnight Rate", 0.16), ("Gasoline Prices", 0.16)],
    21: [("Interest Rate", 76.46), ("High", 9.25),
         ("Overnight Deposit", 1.64), ("Forex Reserves", 1.59), ("Exports", 1.35)],
    30: [("Cash Remittances", 20.16), ("High", 13.21),
         ("Balance of Trade", 12.63), ("Low", 11.00), ("Interest Rate", 9.44)],
}

TFT_DECODER = {
    1:  [("High", 90.17), ("Low", 1.61), ("Forex Reserves", 1.39),
         ("Open", 1.12), ("Overnight Rate", 0.93)],
    3:  [("High", 78.40), ("Low", 3.62), ("Gasoline Prices", 2.68),
         ("Rel. Time Index", 2.44), ("Open", 2.07)],
    7:  [("High", 58.58), ("Imports", 6.76), ("Low", 6.01),
         ("Gasoline Prices", 4.37), ("Overnight Rate", 4.19)],
    14: [("High", 63.46), ("Low", 6.00), ("Imports", 4.86),
         ("Gasoline Prices", 3.48), ("Overnight Deposit", 3.41)],
    21: [("High", 50.53), ("Gasoline Prices", 7.22), ("Inflation Rate", 5.99),
         ("Low", 5.91), ("Forex Reserves", 5.87)],
    30: [("High", 50.53), ("Gasoline Prices", 7.22), ("Forex Reserves", 5.87),
         ("Inflation Rate", 5.99), ("Balance of Trade", 5.00)],
}

REGIME_TEXT = {
    1:  "**1-Day:** Both models rely on **cost-push indicators** — Core Inflation and Gasoline Prices. Daily exchange rate fluctuations are highly sensitive to real-time inflationary pressures.",
    3:  "**3-Day:** A clear shift toward **monetary policy** — the Interest Rate dominates TFT (95.20%) and tops N-BEATSx (40.40%). The cost of borrowing becomes the primary medium-term currency anchor.",
    7:  "**7-Day:** Models briefly diverge. N-BEATSx deepens reliance on **Inflation Rate (42.56%)** and **Overnight Deposit Rate (26.44%)**. TFT pivots to structural market boundaries — historical High (70.00%).",
    14: "**14-Day:** TFT overwhelmingly anchors on **Interest Rate (97.82%)**. N-BEATSx maintains a distributed macroeconomic profile: Inflation Rate (33.62%) and Gasoline Prices (23.01%).",
    21: "**21-Day:** A structural pivot begins — **Cash Remittances** emerge as a top-3 feature in N-BEATSx (17.10%), signaling that international capital inflows carry more weight than short-term domestic volatility.",
    30: "**30-Day:** A complete **regime shift** to international trade dynamics. Both models independently prioritize **Balance of Trade** and **Cash Remittances** — the true fundamental determinants of monthly USD-PHP movements.",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_tft(horizon: int):
    path = PRED_DIR / f"tft_{horizon}day.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


# ── Chart builders ────────────────────────────────────────────────────────────
def heatmap_fig(metric: str, low_is_good: bool = True) -> go.Figure:
    data  = METRICS[metric]
    z     = [data[m] for m in MODELS]
    text  = [[f"{v:.4f}" for v in row] for row in z]
    scale = "RdYlGn_r" if low_is_good else "RdYlGn"
    fig   = go.Figure(go.Heatmap(
        z=z, x=H_LABELS, y=MODELS,
        colorscale=scale,
        text=text, texttemplate="%{text}",
        textfont={"size": 12, "color": "black"},
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{metric} Heatmap — Models × Horizons",
        height=300, margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Forecast Horizon",
    )
    return fig


def bar_fig(items: list, title: str, color: str) -> go.Figure:
    rev      = list(reversed(items))
    features = [i[0] for i in rev]
    values   = [i[1] for i in rev]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=color,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, height=280,
        margin=dict(l=0, r=70, t=45, b=0),
        xaxis=dict(title="Importance (%)", range=[0, max(values) * 1.3]),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.title("USD-PHP Forecasting")
st.sidebar.caption("Thesis Defense Dashboard")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate to",
    ["📊  Page 1 — Error Metrics",
     "🔍  Page 2 — Feature Importance",
     "📈  Page 3 — Line Plots"],
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — 4.1.1  Error Metrics
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊  Page 1 — Error Metrics":
    st.title("4.1.1 Comparative Analysis of Standard Error Metrics")
    st.markdown(
        "MAE, MSE, MAPE, and R² heatmaps across all six forecasting horizons. "
        "**Green = better · Red = worse.**"
    )
    st.divider()

    st.subheader("Mean Absolute Error (MAE)")
    st.plotly_chart(heatmap_fig("MAE", low_is_good=True), use_container_width=True)
    st.caption(
        "The Constrained Ensemble maintains the lowest absolute error at every horizon "
        "(0.1422 at 1-day to 0.7965 at 30-day). TFT suffers a catastrophic spike at 14 days (1.3390)."
    )
    st.divider()

    st.subheader("Mean Squared Error (MSE)")
    st.plotly_chart(heatmap_fig("MSE", low_is_good=True), use_container_width=True)
    st.caption(
        "MSE penalizes large deviations heavily. TFT peaks at 2.5037 at the 14-day horizon. "
        "The Ensemble maintains a controlled 0.2205 at 14 days and 0.4477 at 21 days."
    )
    st.divider()

    st.subheader("Mean Absolute Percentage Error (MAPE)")
    st.plotly_chart(heatmap_fig("MAPE", low_is_good=True), use_container_width=True)
    st.caption(
        "The Ensemble keeps relative error remarkably low (0.0097) even at 21 days. "
        "TFT jumps sharply to 0.0234 at 14 days."
    )
    st.divider()

    st.subheader("R² Score")
    st.plotly_chart(heatmap_fig("R²", low_is_good=False), use_container_width=True)
    st.caption(
        "TFT's R² plunges to −1.3225 at 14 days — worse than the historical mean. "
        "The Ensemble preserves strong explanatory power to 14 days (0.7953) and 21 days (0.5643). "
        "By 30 days every model records a negative R²."
    )
    st.divider()

    st.subheader("Constrained Ensemble Weight Allocations")
    ew_df = pd.DataFrame(
        [(f"{h}-Day", w[0], w[1]) for h, w in ENSEMBLE_WEIGHTS.items()],
        columns=["Horizon", "N-BEATSx", "TFT"],
    )
    fig_ew = go.Figure()
    fig_ew.add_bar(x=ew_df["Horizon"], y=ew_df["TFT"],       name="TFT",      marker_color="#1f77b4")
    fig_ew.add_bar(x=ew_df["Horizon"], y=ew_df["N-BEATSx"],  name="N-BEATSx", marker_color="#ff7f0e")
    fig_ew.update_layout(
        barmode="stack",
        title="Ensemble Weight Distribution per Horizon",
        xaxis_title="Horizon", yaxis_title="Allocated Weight",
        legend=dict(orientation="h", y=1.1), height=340,
    )
    st.plotly_chart(fig_ew, use_container_width=True)
    st.caption(
        "TFT dominates at short horizons (59.87% at 1-day, 86.20% at 7-day). "
        "N-BEATSx takes complete control at extended horizons (95.77% at 14-day, 100% at 21-day)."
    )

    with st.expander("Full Metrics Table"):
        for metric in ["MAE", "MSE", "MAPE", "R²"]:
            st.markdown(f"**{metric}**")
            df_m = pd.DataFrame(METRICS[metric], index=H_LABELS).T
            st.dataframe(df_m.style.format("{:.4f}"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — 4.1.2  Feature Importance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Page 2 — Feature Importance":
    st.title("4.1.2 Comparative Analysis of Feature Importance")
    st.markdown(
        "N-BEATSx **LOFO** ablation scores vs. TFT **Variable Selection Network** (VSN) weights. "
        "Slide across horizons to trace how each model shifts its economic reasoning."
    )
    st.divider()

    horizon = st.select_slider(
        "Forecast Horizon",
        options=HORIZONS,
        format_func=lambda x: f"{x}-Day",
    )
    st.divider()

    col_nb, col_tft = st.columns(2)

    with col_nb:
        st.subheader("N-BEATSx — LOFO Importance")
        st.plotly_chart(
            bar_fig(NBEATSX_LOFO[horizon], f"Top Features · {horizon}-Day", "#ff7f0e"),
            use_container_width=True,
        )

    with col_tft:
        st.subheader("TFT — Variable Selection Network")
        enc_tab, dec_tab = st.tabs(["Encoder (Historical)", "Decoder (Future Known)"])
        with enc_tab:
            st.plotly_chart(
                bar_fig(TFT_ENCODER[horizon], f"Encoder VSN · {horizon}-Day", "#1f77b4"),
                use_container_width=True,
            )
        with dec_tab:
            st.plotly_chart(
                bar_fig(TFT_DECODER[horizon], f"Decoder VSN · {horizon}-Day", "#2ca02c"),
                use_container_width=True,
            )

    st.divider()
    st.info(REGIME_TEXT[horizon])

    with st.expander("Top Feature per Horizon — Summary"):
        st.markdown("**N-BEATSx — #1 Feature**")
        st.dataframe(
            pd.DataFrame(
                [(f"{h}-Day", NBEATSX_LOFO[h][0][0], f"{NBEATSX_LOFO[h][0][1]:.2f}%")
                 for h in HORIZONS],
                columns=["Horizon", "Top Feature", "Importance"],
            ),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**TFT Encoder — #1 Feature**")
        st.dataframe(
            pd.DataFrame(
                [(f"{h}-Day", TFT_ENCODER[h][0][0], f"{TFT_ENCODER[h][0][1]:.2f}%")
                 for h in HORIZONS],
                columns=["Horizon", "Top Feature", "Weight"],
            ),
            hide_index=True, use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Line Plots
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Page 3 — Line Plots":
    st.title("Actual vs. Predicted — Line Plots")
    st.markdown("Inspect the full forecast trajectory or step through the simulation day-by-day.")

    col_h, col_m = st.columns(2)
    with col_h:
        h_choice = st.selectbox("Forecast Horizon", HORIZONS, format_func=lambda x: f"{x}-Day")
    with col_m:
        st.selectbox("Model", ["TFT"])

    df = load_tft(h_choice)
    if df is None:
        st.error(f"Missing `predictions/tft_{h_choice}day.csv`.")
        st.stop()

    dates     = df["Date"].values
    actual    = df["Actual"].values
    predicted = df["Predicted"].values
    n         = len(dates)

    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(mean_absolute_percentage_error(actual, predicted) * 100)
    r2   = float(r2_score(actual, predicted))
    da   = float(np.mean((np.diff(actual) > 0) == (np.diff(predicted) > 0)) * 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE",           f"{mae:.4f}")
    c2.metric("RMSE",          f"{rmse:.4f}")
    c3.metric("MAPE",          f"{mape:.2f}%")
    c4.metric("R²",       f"{r2:.4f}")
    c5.metric("Dir. Accuracy", f"{da:.1f}%")
    st.divider()

    tab_full, tab_sim = st.tabs(["📊 Full Chart", "▶ Animated Simulation"])

    with tab_full:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=actual,    name="Actual",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=predicted, name="TFT Predicted",
                                 line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig.update_layout(
            title=f"TFT — {h_choice}-Day Horizon: Actual vs. Predicted",
            xaxis_title="Date", yaxis_title="USD-PHP (Close)",
            hovermode="x unified", legend=dict(orientation="h", y=1.02), height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_sim:
        min_window = min(30, n)

        if "sim_idx" not in st.session_state:
            st.session_state.sim_idx = min_window
        if "playing" not in st.session_state:
            st.session_state.playing = False
        if "sim_key" not in st.session_state:
            st.session_state.sim_key = h_choice
        if st.session_state.sim_key != h_choice:
            st.session_state.sim_idx = min_window
            st.session_state.playing = False
            st.session_state.sim_key = h_choice

        ctrl_l, ctrl_r = st.columns([1, 3])
        with ctrl_l:
            if st.button("⏸ Pause" if st.session_state.playing else "▶ Play"):
                st.session_state.playing = not st.session_state.playing
            if st.button("⏮ Reset"):
                st.session_state.sim_idx = min_window
                st.session_state.playing = False
        with ctrl_r:
            speed = st.slider("Speed (steps/sec)", 1, 20, 5)

        frame = st.slider("Step", min_value=min_window, max_value=n,
                          value=st.session_state.sim_idx)
        st.session_state.sim_idx = frame
        end = frame

        run_mae = float(mean_absolute_error(actual[:end], predicted[:end]))
        run_r2  = float(r2_score(actual[:end], predicted[:end]))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=dates, y=actual, name="Full Actual (preview)",
                                  line=dict(color="rgba(31,119,180,0.15)", width=1),
                                  hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=dates[:end], y=actual[:end],    name="Actual",
                                  line=dict(color="#1f77b4", width=2)))
        fig3.add_trace(go.Scatter(x=dates[:end], y=predicted[:end], name="TFT Predicted",
                                  line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig3.add_trace(go.Scatter(x=[dates[end - 1]], y=[actual[end - 1]],
                                  mode="markers", marker=dict(size=10, color="#1f77b4"),
                                  showlegend=False, hoverinfo="skip"))
        fig3.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text=f"MAE: {run_mae:.4f}  |  R²: {run_r2:.4f}",
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray",
            borderwidth=1, font=dict(size=11, family="monospace"),
        )
        cur_date = pd.Timestamp(dates[end - 1]).strftime("%Y-%m-%d")
        fig3.update_layout(
            title=f"Day {end} / {n}  ·  {cur_date}",
            xaxis_title="Date", yaxis_title="USD-PHP (Close)",
            hovermode="x unified", legend=dict(orientation="h", y=1.02), height=500,
        )
        st.plotly_chart(fig3, use_container_width=True)

        a1, a2 = st.columns(2)
        a1.metric("Actual",        f"{actual[end - 1]:.4f}")
        a2.metric("TFT Prediction", f"{predicted[end - 1]:.4f}",
                  delta=f"{predicted[end - 1] - actual[end - 1]:+.4f}")

        if st.session_state.playing:
            if st.session_state.sim_idx < n:
                time.sleep(1.0 / speed)
                st.session_state.sim_idx = min(st.session_state.sim_idx + 1, n)
                st.rerun()
            else:
                st.session_state.playing = False
                st.rerun()
