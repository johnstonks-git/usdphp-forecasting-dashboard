import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PRED_DIR = BASE_DIR / "predictions"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="USD-PHP Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

HORIZONS = [1, 3, 7, 14, 21, 30]
ENSEMBLE_HORIZONS = {1, 3, 7}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_tft(horizon: int) -> pd.DataFrame | None:
    path = PRED_DIR / f"tft_{horizon}day.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_nbeatsx_forecasts(horizon: int) -> pd.DataFrame | None:
    """Per-date NBEATSx predictions (Date, Actual, Predicted)."""
    path = PRED_DIR / f"nbeatsx_{horizon}day.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_nbeatsx_optuna(horizon: int) -> pd.DataFrame | None:
    """Optuna trial metrics for NBEATSx (used in Model Comparison when per-date file is absent)."""
    path = PRED_DIR / f"nbeatsx_optuna_{horizon}day.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def build_ensemble(horizon: int):
    tft = load_tft(horizon)
    nb = load_nbeatsx_forecasts(horizon)
    if tft is None or nb is None:
        return None, None

    merged = pd.merge(
        tft[["Date", "Actual", "Predicted"]].rename(columns={"Predicted": "TFT"}),
        nb[["Date", "Predicted"]].rename(columns={"Predicted": "NBEATSx"}),
        on="Date",
        how="inner",
    ).reset_index(drop=True)

    X = merged[["NBEATSx", "TFT"]].values
    y = merged["Actual"].values

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    w1u, w2u = lr.coef_
    bu = lr.intercept_
    merged["Unconstrained"] = lr.predict(X)

    def obj(w):
        return np.mean((y - w[0] * X[:, 0] - w[1] * X[:, 1]) ** 2)

    res = minimize(
        obj,
        [0.5, 0.5],
        method="SLSQP",
        bounds=((0, 1), (0, 1)),
        constraints={"type": "eq", "fun": lambda w: w[0] + w[1] - 1},
    )
    w1c, w2c = res.x
    merged["Constrained"] = w1c * X[:, 0] + w2c * X[:, 1]

    weights = {"unconstrained": (w1u, w2u, bu), "constrained": (w1c, w2c)}
    return merged, weights


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(mean_absolute_percentage_error(actual, predicted) * 100)
    r2 = float(r2_score(actual, predicted))
    da = float(np.mean((np.diff(actual) > 0) == (np.diff(predicted) > 0)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2, "DA (%)": da}


def get_model_series(model_name, tft_df, nb_df, ens_df):
    """Return (dates, actual, predicted) arrays for the selected model."""
    if model_name == "TFT":
        return tft_df["Date"].values, tft_df["Actual"].values, tft_df["Predicted"].values
    if model_name == "N-BEATSx":
        return nb_df["Date"].values, nb_df["Actual"].values, nb_df["Predicted"].values
    if model_name == "Unconstrained Ensemble":
        return ens_df["Date"].values, ens_df["Actual"].values, ens_df["Unconstrained"].values
    if model_name == "Constrained Ensemble":
        return ens_df["Date"].values, ens_df["Actual"].values, ens_df["Constrained"].values


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.title("📈 USD-PHP Exchange Rate Forecasting Dashboard")
st.caption("N-BEATSx  ·  Temporal Fusion Transformer  ·  Ensemble — Multi-Horizon Performance Demo")

with st.sidebar:
    st.header("Controls")
    horizon = st.selectbox("Forecast Horizon (days)", HORIZONS)

    tft_df = load_tft(horizon)
    nb_df = load_nbeatsx_forecasts(horizon) if horizon in ENSEMBLE_HORIZONS else None
    ens_df, ens_weights = (
        build_ensemble(horizon) if horizon in ENSEMBLE_HORIZONS else (None, None)
    )

    available_models: list[str] = []
    if tft_df is not None:
        available_models.append("TFT")
    if nb_df is not None:
        available_models += ["N-BEATSx", "Unconstrained Ensemble", "Constrained Ensemble"]

    if not available_models:
        st.error(f"No prediction files found for {horizon}-day horizon.")
        st.stop()

    model = st.selectbox("Model", available_models)

    st.divider()
    st.markdown("**Prediction file status:**")
    for h in HORIZONS:
        ok = (PRED_DIR / f"tft_{h}day.csv").exists()
        st.caption(f"{'✅' if ok else '❌'} TFT {h}-day")
    for h in [1, 3, 7]:
        ok = (PRED_DIR / f"nbeatsx_{h}day.csv").exists()
        st.caption(f"{'✅' if ok else '⏳'} N-BEATSx {h}-day{'  ← pending' if not ok else ''}")

if tft_df is None:
    st.error(
        f"**Missing:** `predictions/tft_{horizon}day.csv`\n\n"
        "Place the TFT prediction CSV in `dashboard/predictions/` and refresh."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Interactive Performance", "🏆 Model Comparison", "▶ Animated Simulation"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    dates, actual, predicted = get_model_series(model, tft_df, nb_df, ens_df)
    metrics = compute_metrics(actual, predicted)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{metrics['MAE']:.4f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
    c3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
    c4.metric("R²", f"{metrics['R²']:.4f}")
    c5.metric("Dir. Accuracy", f"{metrics['DA (%)']:.1f}%")

    st.divider()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="#1f77b4", width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            name=f"{model} Predicted",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{model} — {horizon}-Day Horizon: Actual vs. Predicted",
        xaxis_title="Date",
        yaxis_title="USD-PHP (Close)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    if model == "Unconstrained Ensemble" and ens_weights:
        w1, w2, b = ens_weights["unconstrained"]
        st.caption(f"Formula: ŷ = {w1:.4f}·N-BEATSx + {w2:.4f}·TFT + {b:.4f}")
    elif model == "Constrained Ensemble" and ens_weights:
        w1, w2 = ens_weights["constrained"]
        st.caption(f"Formula: ŷ = {w1:.4f}·N-BEATSx + {w2:.4f}·TFT  (weights sum = 1)")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Model Comparison — {horizon}-Day Horizon")

    rows = []

    if tft_df is not None:
        rows.append(
            {
                "Model": "TFT",
                **compute_metrics(tft_df["Actual"].values, tft_df["Predicted"].values),
            }
        )

    if nb_df is not None:
        rows.append(
            {
                "Model": "N-BEATSx",
                **compute_metrics(nb_df["Actual"].values, nb_df["Predicted"].values),
            }
        )
    elif horizon in ENSEMBLE_HORIZONS:
        # Use best Optuna trial as proxy for N-BEATSx performance
        optuna_df = load_nbeatsx_optuna(horizon)
        if optuna_df is not None:
            best = optuna_df.sort_values("Test_MAE").iloc[0]
            rows.append(
                {
                    "Model": "N-BEATSx (best trial)*",
                    "MAE": best["Test_MAE"],
                    "RMSE": float(np.sqrt(best["Test_MSE"])),
                    "MAPE (%)": best["Test_MAPE"] * 100,
                    "R²": best["Test_R2"],
                    "DA (%)": best["Test_DA"],
                }
            )

    if ens_df is not None:
        rows.append(
            {
                "Model": "Unconstrained Ensemble",
                **compute_metrics(ens_df["Actual"].values, ens_df["Unconstrained"].values),
            }
        )
        rows.append(
            {
                "Model": "Constrained Ensemble",
                **compute_metrics(ens_df["Actual"].values, ens_df["Constrained"].values),
            }
        )

    if not rows:
        st.info("No model data available for this horizon yet.")
    else:
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df.set_index("Model").style.format("{:.4f}"), use_container_width=True)

        if any("best trial" in r["Model"] for r in rows):
            st.caption(
                "\\* N-BEATSx metrics shown from best Optuna trial. "
                "Add `nbeatsx_Xday.csv` to predictions/ for per-date forecast comparison."
            )

        st.divider()
        metric_choice = st.radio(
            "Metric to visualize",
            ["MAE", "RMSE", "MAPE (%)", "R²", "DA (%)"],
            horizontal=True,
            key="tab2_metric",
        )
        fig2 = px.bar(
            comp_df,
            x="Model",
            y=metric_choice,
            color="Model",
            text_auto=".4f",
            title=f"{metric_choice} — {horizon}-Day Horizon",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    if horizon not in ENSEMBLE_HORIZONS:
        st.info("N-BEATSx and Ensemble models are only available for 1, 3, and 7-day horizons.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Animated Simulation
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Animated Simulation — {model} ({horizon}-Day Horizon)")
    st.caption("Step through the test period to see how the model performed day-by-day.")

    dates_sim, actual_sim, predicted_sim = get_model_series(model, tft_df, nb_df, ens_df)
    n = len(dates_sim)
    min_window = min(30, n)

    if "sim_idx" not in st.session_state:
        st.session_state.sim_idx = min_window
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "sim_model_key" not in st.session_state:
        st.session_state.sim_model_key = (model, horizon)

    # Reset playhead when model/horizon changes
    if st.session_state.sim_model_key != (model, horizon):
        st.session_state.sim_idx = min_window
        st.session_state.playing = False
        st.session_state.sim_model_key = (model, horizon)

    ctrl_l, ctrl_r = st.columns([1, 3])
    with ctrl_l:
        play_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
        if st.button(play_label, key="play_btn"):
            st.session_state.playing = not st.session_state.playing
        if st.button("⏮ Reset", key="reset_btn"):
            st.session_state.sim_idx = min_window
            st.session_state.playing = False
    with ctrl_r:
        speed = st.slider("Speed (steps/sec)", 1, 20, 5, key="speed_slider")

    frame_idx = st.slider(
        "Date index",
        min_value=min_window,
        max_value=n,
        value=st.session_state.sim_idx,
        key="sim_frame_slider",
    )
    st.session_state.sim_idx = frame_idx

    def make_sim_frame(end: int) -> go.Figure:
        fig3 = go.Figure()
        # Faded full actual
        fig3.add_trace(
            go.Scatter(
                x=dates_sim,
                y=actual_sim,
                name="Full Actual (preview)",
                line=dict(color="rgba(31,119,180,0.15)", width=1),
                hoverinfo="skip",
            )
        )
        # Revealed actual
        fig3.add_trace(
            go.Scatter(
                x=dates_sim[:end],
                y=actual_sim[:end],
                name="Actual",
                line=dict(color="#1f77b4", width=2),
            )
        )
        # Predictions revealed so far
        fig3.add_trace(
            go.Scatter(
                x=dates_sim[:end],
                y=predicted_sim[:end],
                name=f"{model}",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )
        # Current point marker
        fig3.add_trace(
            go.Scatter(
                x=[dates_sim[end - 1]],
                y=[actual_sim[end - 1]],
                mode="markers",
                marker=dict(size=10, color="#1f77b4", symbol="circle"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Running metrics annotation
        run_m = compute_metrics(actual_sim[:end], predicted_sim[:end])
        note = (
            f"MAE: {run_m['MAE']:.4f}  |  RMSE: {run_m['RMSE']:.4f}<br>"
            f"MAPE: {run_m['MAPE (%)']:.2f}%  |  R²: {run_m['R²']:.4f}  |  DA: {run_m['DA (%)']:.1f}%"
        )
        fig3.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            text=note,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11, family="monospace"),
        )
        cur_date = pd.Timestamp(dates_sim[end - 1]).strftime("%Y-%m-%d")
        fig3.update_layout(
            title=f"Day {end} / {n}  ·  {cur_date}",
            xaxis_title="Date",
            yaxis_title="USD-PHP (Close)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
            height=500,
        )
        return fig3

    chart_ph = st.empty()
    chart_ph.plotly_chart(make_sim_frame(st.session_state.sim_idx), use_container_width=True)

    cur = st.session_state.sim_idx - 1
    m1, m2 = st.columns(2)
    m1.metric("Actual (current day)", f"{actual_sim[cur]:.4f}")
    m2.metric(
        f"{model} Prediction",
        f"{predicted_sim[cur]:.4f}",
        delta=f"{predicted_sim[cur] - actual_sim[cur]:+.4f}",
    )

    # Auto-advance when playing
    if st.session_state.playing:
        if st.session_state.sim_idx < n:
            time.sleep(1.0 / speed)
            st.session_state.sim_idx = min(st.session_state.sim_idx + 1, n)
            st.rerun()
        else:
            st.session_state.playing = False
            st.rerun()
