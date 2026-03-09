import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Currency Forecasting",
    page_icon="💱",
    layout="wide"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    stat = pd.read_csv("data/predictions_stat.csv", parse_dates=["date"])
    ml   = pd.read_csv("data/predictions_ml.csv",   parse_dates=["date"])
    dl   = pd.read_csv("data/predictions_dl.csv",   parse_dates=["date"])
    ens  = pd.read_csv("data/predictions_ensemble.csv", parse_dates=["date"])
    return stat, ml, dl, ens

stat_df, ml_df, dl_df, ens_df = load_data()

# ── Metrics function ───────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot
    da   = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2, "DA (%)": da}

# ── Plot function ──────────────────────────────────────────────────────────────
def plot_forecast(dates, actual, predictions, title, currency):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actual, color="#2196F3", linewidth=1.0, label="Actual", zorder=5)
    colors = ["#FF5722", "#FFA726", "#4CAF50", "#9C27B0", "#00BCD4"]
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(dates, pred, color=colors[i % len(colors)],
                linewidth=0.9, linestyle="--", label=name, alpha=0.85)
    ax.set_title(f"{currency} — {title}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Exchange Rate")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.shields.io/badge/Currency-Forecasting-2196F3?style=for-the-badge")
    st.markdown("---")
    st.markdown("### 💱 Currency Forecasting")
    st.markdown("End-to-end forecasting system for **SGD/USD** and **CNY/USD** using 12 models across 4 model families.")
    st.markdown("---")
    currency = st.radio("**Select Currency**", ["SGD/USD", "CNY/USD"], index=0)
    cur = "sgd" if currency == "SGD/USD" else "cny"
    st.markdown("---")
    st.markdown("**Model Families**")
    st.markdown("- 📊 Statistical (8 models)")
    st.markdown("- 🤖 Machine Learning (5 models)")
    st.markdown("- 🧠 Deep Learning (3 models)")
    st.markdown("- 🔀 Ensemble (2 models)")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- Period: 2000–2019")
    st.markdown("- Frequency: Daily")
    st.markdown("- Observations: 5,217")
    st.markdown("---")
    st.markdown("Built by **Akakinad**")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Akakinad-181717?logo=github)](https://github.com/Akakinad)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.title("💱 Currency Exchange Rate Forecasting")
st.markdown(f"### Forecasting **{currency}** — 2016 to 2019 Test Period")
st.markdown("---")

# ── Top KPI cards ──────────────────────────────────────────────────────────────
ens_actual  = ens_df[f"actual_{cur}"].values
ens_pred    = ens_df[f"ensemble_{cur}_weighted"].values
best_metrics = compute_metrics(ens_actual, ens_pred)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏆 Best Model", "Weighted Ensemble")
col2.metric("📉 RMSE", f"{best_metrics['RMSE']:.5f}")
col3.metric("📈 R²", f"{best_metrics['R²']:.4f}")
col4.metric("🎯 Directional Acc.", f"{best_metrics['DA (%)']:.2f}%")
col5.metric("📊 Total Models", "12")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Statistical Models",
    "🤖 Machine Learning",
    "🧠 Deep Learning",
    "🔀 Ensemble",
    "🏆 Final Comparison"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATISTICAL MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Statistical Models")
    st.markdown("Classical time series models applied to exchange rate forecasting.")

    stat_models = {
        "Naïve": f"naive_{cur}",
        "Holt-Winters": f"hw_{cur}",
        "ARIMA": f"arima_{cur}",
        "SARIMA": f"sarima_{cur}",
        "SARIMAX": f"sarimax_{cur}",
        "VAR": f"var_{cur}",
        "VECM": f"vecm_{cur}",
        "Prophet": f"prophet_{cur}",
    }

    selected_stat = st.multiselect(
        "Select models to display",
        list(stat_models.keys()),
        default=["Naïve", "SARIMAX", "Prophet"]
    )

    if selected_stat:
        preds = {m: stat_df[stat_models[m]].values for m in selected_stat}
        fig = plot_forecast(stat_df["date"], stat_df[f"actual_{cur}"], preds,
                           "Statistical Model Forecasts", currency)
        st.pyplot(fig)

        st.markdown("#### Metrics")
        rows = []
        for m in stat_models:
            met = compute_metrics(stat_df[f"actual_{cur}"], stat_df[stat_models[m]])
            rows.append({"Model": m, **met})
        df_metrics = pd.DataFrame(rows).set_index("Model")
        df_metrics = df_metrics.style.format({
            "RMSE": "{:.6f}", "MAE": "{:.6f}",
            "MAPE (%)": "{:.4f}", "R²": "{:.4f}", "DA (%)": "{:.2f}"
        }).highlight_min(subset=["RMSE", "MAE"], color="#d4edda") \
          .highlight_max(subset=["R²", "DA (%)"], color="#d4edda")
        st.dataframe(df_metrics, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Machine Learning Models")
    st.markdown("Tree-based and kernel models trained on 34 engineered features.")

    ml_models = {
        "Decision Tree": f"dt_{cur}",
        "Random Forest": f"rf_{cur}",
        "SVR": f"svr_{cur}",
        "XGBoost": f"xgb_{cur}",
        "LightGBM": f"lgbm_{cur}",
    }

    selected_ml = st.multiselect(
        "Select models to display",
        list(ml_models.keys()),
        default=["XGBoost", "LightGBM", "SVR"]
    )

    if selected_ml:
        preds = {m: ml_df[ml_models[m]].values for m in selected_ml}
        fig = plot_forecast(ml_df["date"], ml_df[f"actual_{cur}"], preds,
                           "Machine Learning Forecasts", currency)
        st.pyplot(fig)

        st.markdown("#### Feature Engineering Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            | Category | Count |
            |---|---|
            | Lag features | 12 |
            | Rolling statistics | 12 |
            | Momentum (ROC) | 4 |
            | Calendar | 4 |
            | Cross-currency | 2 |
            | **Total** | **34** |
            """)
        with col2:
            st.markdown("#### Metrics")
            rows = []
            for m in ml_models:
                met = compute_metrics(ml_df[f"actual_{cur}"], ml_df[ml_models[m]])
                rows.append({"Model": m, **met})
            df_metrics = pd.DataFrame(rows).set_index("Model")
            df_metrics = df_metrics.style.format({
                "RMSE": "{:.6f}", "MAE": "{:.6f}",
                "MAPE (%)": "{:.4f}", "R²": "{:.4f}", "DA (%)": "{:.2f}"
            }).highlight_min(subset=["RMSE", "MAE"], color="#d4edda") \
              .highlight_max(subset=["R²", "DA (%)"], color="#d4edda")
            st.dataframe(df_metrics, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 Deep Learning Models")
    st.markdown("Sequence models trained on 60-day windows of historical rates.")

    dl_models = {
        "LSTM": f"lstm_{cur}",
        "GRU": f"gru_{cur}",
        "Transformer": f"transformer_{cur}",
    }

    selected_dl = st.multiselect(
        "Select models to display",
        list(dl_models.keys()),
        default=["LSTM", "GRU", "Transformer"]
    )

    if selected_dl:
        preds = {m: dl_df[dl_models[m]].values for m in selected_dl}
        fig = plot_forecast(dl_df["date"], dl_df[f"actual_{cur}"], preds,
                           "Deep Learning Forecasts", currency)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.markdown("**LSTM**\n\n`LSTM(64) → Dropout → LSTM(32) → Dense(1)`\n\nParams: 29,345")
        col2.markdown("**GRU**\n\n`GRU(64) → Dropout → GRU(32) → Dense(1)`\n\nParams: 22,305")
        col3.markdown("**Transformer**\n\n`MHA(heads=4) → LayerNorm → FF(64) → Dense(1)`\n\nParams: 9,697")

        st.markdown("#### Metrics")
        rows = []
        for m in dl_models:
            met = compute_metrics(dl_df[f"actual_{cur}"], dl_df[dl_models[m]])
            rows.append({"Model": m, **met})
        df_metrics = pd.DataFrame(rows).set_index("Model")
        df_metrics = df_metrics.style.format({
            "RMSE": "{:.6f}", "MAE": "{:.6f}",
            "MAPE (%)": "{:.4f}", "R²": "{:.4f}", "DA (%)": "{:.2f}"
        }).highlight_min(subset=["RMSE", "MAE"], color="#d4edda") \
          .highlight_max(subset=["R²", "DA (%)"], color="#d4edda")
        st.dataframe(df_metrics, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔀 Ensemble Models")
    st.markdown("Combining XGBoost, LightGBM and GRU predictions for superior accuracy.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Simple Ensemble**
        ```
        Final = (XGBoost + LightGBM + GRU) / 3
        ```
        Equal weights for all models.
        """)
    with col2:
        st.markdown("""
        **Weighted Ensemble**
        Weights inversely proportional to test RMSE:

        | Model | SGD Weight | CNY Weight |
        |---|---|---|
        | XGBoost | 0.357 | 0.291 |
        | LightGBM | 0.359 | 0.301 |
        | GRU | 0.284 | 0.408 |
        """)

    ens_models = {
        "Simple Ensemble": f"ensemble_{cur}_simple",
        "Weighted Ensemble": f"ensemble_{cur}_weighted",
        "XGBoost": f"xgb_{cur}",
        "LightGBM": f"lgbm_{cur}",
        "GRU": f"gru_{cur}",
    }

    preds = {m: ens_df[ens_models[m]].values for m in ens_models}
    fig = plot_forecast(ens_df["date"], ens_df[f"actual_{cur}"], preds,
                       "Ensemble Forecasts", currency)
    st.pyplot(fig)

    st.markdown("#### Metrics")
    rows = []
    for m in ens_models:
        met = compute_metrics(ens_df[f"actual_{cur}"], ens_df[ens_models[m]])
        rows.append({"Model": m, **met})
    df_metrics = pd.DataFrame(rows).set_index("Model")
    df_metrics = df_metrics.style.format({
        "RMSE": "{:.6f}", "MAE": "{:.6f}",
        "MAPE (%)": "{:.4f}", "R²": "{:.4f}", "DA (%)": "{:.2f}"
    }).highlight_min(subset=["RMSE", "MAE"], color="#d4edda") \
      .highlight_max(subset=["R²", "DA (%)"], color="#d4edda")
    st.dataframe(df_metrics, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🏆 Final Model Comparison — All 12 Models")

    all_models = {
        "Naïve": (stat_df, f"naive_{cur}"),
        "Holt-Winters": (stat_df, f"hw_{cur}"),
        "ARIMA": (stat_df, f"arima_{cur}"),
        "SARIMA": (stat_df, f"sarima_{cur}"),
        "SARIMAX": (stat_df, f"sarimax_{cur}"),
        "VAR": (stat_df, f"var_{cur}"),
        "VECM": (stat_df, f"vecm_{cur}"),
        "Prophet": (stat_df, f"prophet_{cur}"),
        "Decision Tree": (ml_df, f"dt_{cur}"),
        "Random Forest": (ml_df, f"rf_{cur}"),
        "SVR": (ml_df, f"svr_{cur}"),
        "XGBoost": (ml_df, f"xgb_{cur}"),
        "LightGBM": (ml_df, f"lgbm_{cur}"),
        "LSTM": (dl_df, f"lstm_{cur}"),
        "GRU": (dl_df, f"gru_{cur}"),
        "Transformer": (dl_df, f"transformer_{cur}"),
        "Simple Ensemble": (ens_df, f"ensemble_{cur}_simple"),
        "Weighted Ensemble": (ens_df, f"ensemble_{cur}_weighted"),
    }

    rows = []
    for model_name, (df, col) in all_models.items():
        met = compute_metrics(df[f"actual_{cur}"], df[col])
        rows.append({"Model": model_name, **met})

    df_all = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df_all.index += 1
    df_all.index.name = "Rank"

    st.markdown(f"#### {currency} — All Models Ranked by RMSE")
    styled = df_all.style.format({
        "RMSE": "{:.6f}", "MAE": "{:.6f}",
        "MAPE (%)": "{:.4f}", "R²": "{:.4f}", "DA (%)": "{:.2f}"
    }).highlight_min(subset=["RMSE", "MAE"], color="#d4edda") \
      .highlight_max(subset=["R²", "DA (%)"], color="#d4edda") \
      .highlight_min(subset=["R²"], color="#f8d7da")
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.markdown("#### RMSE Comparison — Bar Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3" if "Ensemble" in m else
              "#4CAF50" if m in ["XGBoost", "LightGBM", "Random Forest", "Decision Tree", "SVR"] else
              "#FF9800" if m in ["LSTM", "GRU", "Transformer"] else
              "#9E9E9E" for m in df_all["Model"]]
    bars = ax.barh(df_all["Model"], df_all["RMSE"], color=colors, edgecolor="white")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title(f"{currency} — Model RMSE Comparison", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, df_all["RMSE"]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Ensemble"),
        Patch(facecolor="#4CAF50", label="Machine Learning"),
        Patch(facecolor="#FF9800", label="Deep Learning"),
        Patch(facecolor="#9E9E9E", label="Statistical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
