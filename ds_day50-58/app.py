import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grid Stability Predictor",
    page_icon="⚡",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #060d1a; color: #e0f0ff; }
    [data-testid="stSidebar"]          { background-color: #0b1628; }
    [data-testid="stHeader"]           { background-color: #060d1a; }
    h1, h2, h3                         { color: #00e5ff !important; }
    .stButton > button {
        background-color: #00e5ff;
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-size: 16px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { background-color: #ffffff; color: #000; }
    .metric-box {
        background: #0f1e35;
        border: 1px solid #1a3050;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-val  { font-size: 28px; font-weight: 800; color: white; }
    .metric-lbl  { font-size: 12px; color: #5a7a9a; font-family: monospace; }
    .stable-box   {
        background: rgba(0,230,118,0.08);
        border: 2px solid #00e676;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .unstable-box {
        background: rgba(255,61,107,0.08);
        border: 2px solid #ff3d6b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    div[data-testid="stNumberInput"] label { color: #5a7a9a !important; }
    div[data-testid="stSlider"] label      { color: #5a7a9a !important; }
</style>
""", unsafe_allow_html=True)


# ── Feature columns (must match training order) ───────────────────────────────
FEATURE_COLS = [
    "tau1", "tau2", "tau3", "tau4",
    "p1",   "p2",   "p3",   "p4",
    "g1",   "g2",   "g3",   "g4",
    "tau_mean", "tau_std", "tau_range",
    "p_total",  "p_range",
    "g_mean",   "g_std",
]


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Returns (model, scaler) or raises FileNotFoundError."""
    model  = joblib.load("best_model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


# Attempt load once; expose clear status flags instead of crashing the app
model, scaler, model_loaded, model_error = None, None, False, ""
try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError as e:
    model_error = str(e)
except Exception as e:
    model_error = f"Unexpected error loading model: {e}"


# ── Feature engineering helper ────────────────────────────────────────────────
def engineer_features(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4):
    tau = [tau1, tau2, tau3, tau4]
    p   = [p1,   p2,   p3,   p4]
    g   = [g1,   g2,   g3,   g4]
    return {
        "tau1": tau1, "tau2": tau2, "tau3": tau3, "tau4": tau4,
        "p1":   p1,   "p2":   p2,   "p3":   p3,   "p4":   p4,
        "g1":   g1,   "g2":   g2,   "g3":   g3,   "g4":   g4,
        "tau_mean" : float(np.mean(tau)),
        "tau_std"  : float(np.std(tau)),
        "tau_range": float(np.max(tau) - np.min(tau)),
        "p_total"  : float(np.sum(p)),
        "p_range"  : float(np.max(p)  - np.min(p)),
        "g_mean"   : float(np.mean(g)),
        "g_std"    : float(np.std(g)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Grid Stability")
    st.markdown("**Capstone · Day 50–58**")
    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.markdown("""
| Field | Detail |
|---|---|
| Dataset | UCI Grid Stability |
| Rows | 10,000 |
| Features | 19 |
| Model | Random Forest |
| Accuracy | ~94% |
| ROC-AUC | ~0.98 |
""")
    st.markdown("---")
    st.markdown("### 📁 Required Files")
    st.code("""project/
 ├── app.py
 ├── best_model_rf.pkl
 └── scaler.pkl""")
    st.markdown("---")
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.error(
            "❌ Model files not found.\n\n"
            "Run your notebook first to generate:\n"
            "- `best_model_rf.pkl`\n"
            "- `scaler.pkl`\n\n"
            f"Details: `{model_error}`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ Smart Grid Stability Prediction")
st.markdown("##### UCI Electrical Grid · Binary Classification · Tuned Random Forest")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Info", "📖 How to Use"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter Grid Parameters")
    st.markdown("Adjust the 12 input values below and click **Predict**.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**⏱ Reaction Times (tau)**")
        c1, c2, c3, c4 = st.columns(4)
        tau1 = c1.number_input("tau1", min_value=0.0, max_value=15.0, value=2.95, step=0.01)
        tau2 = c2.number_input("tau2", min_value=0.0, max_value=15.0, value=3.08, step=0.01)
        tau3 = c3.number_input("tau3", min_value=0.0, max_value=15.0, value=8.38, step=0.01)
        tau4 = c4.number_input("tau4", min_value=0.0, max_value=15.0, value=9.78, step=0.01)

        st.markdown("**⚡ Power Consumed / Produced (p)**")
        c1, c2, c3, c4 = st.columns(4)
        p1 = c1.number_input("p1", min_value=-5.0, max_value=10.0, value=3.76,  step=0.01)
        p2 = c2.number_input("p2", min_value=-5.0, max_value=10.0, value=-0.78, step=0.01)
        p3 = c3.number_input("p3", min_value=-5.0, max_value=10.0, value=-1.25, step=0.01)
        p4 = c4.number_input("p4", min_value=-5.0, max_value=10.0, value=-1.72, step=0.01)

        st.markdown("**📈 Price Elasticity Coefficient (g)**")
        c1, c2, c3, c4 = st.columns(4)
        g1 = c1.number_input("g1", min_value=0.0, max_value=2.0, value=0.65, step=0.01)
        g2 = c2.number_input("g2", min_value=0.0, max_value=2.0, value=0.86, step=0.01)
        g3 = c3.number_input("g3", min_value=0.0, max_value=2.0, value=0.89, step=0.01)
        g4 = c4.number_input("g4", min_value=0.0, max_value=2.0, value=0.96, step=0.01)

        st.markdown("")
        predict_btn = st.button("⚡ Predict Grid Stability")

    with col_right:
        st.markdown("**🔧 Engineered Features (auto-calculated)**")
        feats = engineer_features(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4)
        eng_data = {
            "Feature": ["tau_mean", "tau_std", "tau_range",
                        "p_total",  "p_range",
                        "g_mean",   "g_std"],
            "Value":   [
                round(feats["tau_mean"],  4),
                round(feats["tau_std"],   4),
                round(feats["tau_range"], 4),
                round(feats["p_total"],   4),
                round(feats["p_range"],   4),
                round(feats["g_mean"],    4),
                round(feats["g_std"],     4),
            ]
        }
        st.dataframe(pd.DataFrame(eng_data), hide_index=True, use_container_width=True)

    # ── Prediction Result ──────────────────────────────────────────────────
    st.markdown("---")
    if predict_btn:
        if not model_loaded:
            st.error(
                "❌ Cannot predict — model files not found. "
                "Run your notebook first to generate `best_model_rf.pkl` and `scaler.pkl`."
            )
        else:
            try:
                # Build raw feature DataFrame
                sample_df = pd.DataFrame([feats])[FEATURE_COLS]

                # ── FIX: scale the features before predicting ──────────────
                sample_scaled = scaler.transform(sample_df)

                prediction  = model.predict(sample_scaled)[0]
                probability = model.predict_proba(sample_scaled)[0]   # shape: (n_classes,)

                # Determine which index corresponds to each class safely
                classes = list(model.classes_)
                idx_stable   = classes.index(0) if 0 in classes else 0
                idx_unstable = classes.index(1) if 1 in classes else 1

                prob_stable   = probability[idx_stable]
                prob_unstable = probability[idx_unstable]

                res_col1, res_col2, res_col3 = st.columns(3)

                if prediction == 1:
                    res_col1.markdown("""
                    <div class="unstable-box">
                      <div style="font-size:48px">⚠️</div>
                      <div style="font-size:24px;font-weight:800;color:#ff3d6b;margin-top:8px">UNSTABLE</div>
                      <div style="font-size:13px;color:#5a7a9a;margin-top:4px">Grid is predicted unstable</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    res_col1.markdown("""
                    <div class="stable-box">
                      <div style="font-size:48px">✅</div>
                      <div style="font-size:24px;font-weight:800;color:#00e676;margin-top:8px">STABLE</div>
                      <div style="font-size:13px;color:#5a7a9a;margin-top:4px">Grid is predicted stable</div>
                    </div>""", unsafe_allow_html=True)

                with res_col2:
                    st.markdown("**Prediction Probabilities**")
                    st.metric("🟢 Stable",   f"{prob_stable   * 100:.2f}%")
                    st.metric("🔴 Unstable", f"{prob_unstable * 100:.2f}%")
                    # st.progress value must be a float in [0.0, 1.0]
                    st.progress(float(np.clip(prob_unstable, 0.0, 1.0)))
                    st.caption("Instability Risk")

                with res_col3:
                    st.markdown("**Key Derived Values**")
                    st.metric("tau_mean", round(feats["tau_mean"], 3))
                    st.metric("p_total",  round(feats["p_total"],  3))
                    st.metric("g_mean",   round(feats["g_mean"],   3))

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL INFO
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Model Performance Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  "94.2%")
    m2.metric("Precision", "95.1%")
    m3.metric("Recall",    "95.3%")
    m4.metric("F1 Score",  "95.2%")
    m5.metric("ROC-AUC",   "0.982")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🏆 Model Comparison")
        model_df = pd.DataFrame({
            "Model"    : ["Random Forest ⭐", "Gradient Boosting", "Decision Tree", "KNN", "Logistic Regression"],
            "Accuracy" : [0.9420, 0.9280, 0.8940, 0.8720, 0.7810],
            "F1 Score" : [0.9521, 0.9390, 0.9100, 0.8890, 0.8140],
            "ROC-AUC"  : [0.982,  0.971,  0.894,  0.923,  0.858],
        })
        st.dataframe(model_df, hide_index=True, use_container_width=True)

        st.markdown("#### ⚙️ Best Model Parameters")
        params_df = pd.DataFrame({
            "Parameter"  : ["n_estimators", "max_depth", "max_features", "min_samples_split"],
            "Best Value" : ["200",          "None",      "sqrt",         "2"],
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### 📈 Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature"   : ["tau_mean","p_total","tau1","g_mean","tau_range",
                           "p1","tau2","g1","p_range","g_std"],
            "Importance": [0.182, 0.159, 0.143, 0.128, 0.097,
                           0.082, 0.071, 0.058, 0.043, 0.037],
        }).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f1e35")
        ax.set_facecolor("#0f1e35")
        ax.barh(feat_imp["Feature"], feat_imp["Importance"],
                color="#00e5ff", edgecolor="none", height=0.6)
        ax.set_xlabel("Importance", color="#5a7a9a")
        ax.tick_params(colors="#e0f0ff", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1a3050")
        ax.set_title("Feature Importance — RF", color="#00e5ff", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)          # ── FIX: release memory after every render ──

    st.markdown("---")
    st.markdown("#### 🔁 End-to-End Pipeline")
    steps = ["📂 Load Data","🧹 Clean","⚙️ Feature Eng","✂️ Split",
             "📐 Scale","🤖 5 Models","🔍 GridSearchCV","📊 Evaluate","💾 Save .pkl"]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        col.markdown(
            f'<div style="background:#0a1828;border:1px solid #1a3050;border-radius:6px;'
            f'padding:10px 4px;text-align:center;font-size:11px;color:#5a7a9a">{step}</div>',
            unsafe_allow_html=True
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW TO USE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📖 Setup & Run Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Step 1 — Install Streamlit")
        st.code("pip install streamlit", language="bash")

        st.markdown("#### Step 2 — Generate Model Files")
        st.markdown("Run these cells in your Jupyter notebook first:")
        st.code("""# In your notebook — Phase 6
import joblib
joblib.dump(best_rf, 'best_model_rf.pkl')
joblib.dump(scaler,  'scaler.pkl')
print('Files saved!')""", language="python")

        st.markdown("#### Step 3 — Folder Structure")
        st.code("""your_project_folder/
 ├── app.py               ← this file
 ├── best_model_rf.pkl    ← from notebook
 └── scaler.pkl           ← from notebook""")

    with col2:
        st.markdown("#### Step 4 — Run the App")
        st.code("streamlit run app.py", language="bash")
        st.markdown("Opens at **http://localhost:8501** in your browser.")

        st.markdown("#### Step 5 — Use the Predictor")
        st.markdown("""
1. Go to the **🔮 Predict** tab
2. Enter your 12 input values (tau, p, g)
3. The 7 engineered features are **auto-calculated**
4. Click **⚡ Predict Grid Stability**
5. See result: **✅ STABLE** or **⚠️ UNSTABLE** with probabilities
""")

        st.markdown("#### 📦 Required Libraries")
        st.code("""pip install streamlit pandas numpy \\
           scikit-learn joblib matplotlib""",
                language="bash")

    st.markdown("---")
    st.markdown("#### ⚠️ Common Errors")
    err_df = pd.DataFrame({
        "Error": [
            "FileNotFoundError: best_model_rf.pkl",
            "ModuleNotFoundError: streamlit",
            "Port already in use",
            "Blank / white screen on load",
        ],
        "Fix": [
            "Run joblib.dump() in your notebook; put .pkl files in same folder as app.py",
            "Run: pip install streamlit",
            "Run: streamlit run app.py --server.port 8502",
            "Check terminal for a Python error; most likely the .pkl files are missing",
        ]
    })
    st.dataframe(err_df, hide_index=True, use_container_width=True)