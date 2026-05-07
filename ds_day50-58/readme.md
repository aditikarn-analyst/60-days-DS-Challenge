# ⚡ Smart Grid Stability Prediction System
### Capstone Project · Day 50–58 · End-to-End Data Science Pipeline

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project builds a complete **end-to-end machine learning pipeline** to predict the stability of an electrical power grid based on participant reaction times, power consumption, and price elasticity coefficients.

A smart grid becomes **unstable** when the system cannot balance power supply and demand in real time. Early prediction of instability can prevent widespread outages.

> **ML Task:** Binary Classification → `stable` / `unstable`  
> **Dataset:** UCI Electrical Grid Stability Simulated Data  
> **Best Model:** Tuned Random Forest → **94.2% Accuracy | 0.982 ROC-AUC**

---

## 🖥️ Live Demo

![App Screenshot](assets/screenshot.png)

```
streamlit run app.py
```
Opens at → **http://localhost:8501**

---

## 📁 Project Structure

```
smart-grid-stability/
│
├── app.py                          ← Streamlit web application
├── requirements.txt                ← Python dependencies
├── .streamlit/
│   └── secrets.toml                ← Streamlit config & secrets
│
├── notebooks/
│   └── Capstone_Day50_58_UCI_GridStability.ipynb   ← Full analysis notebook
│
├── models/
│   ├── best_model_rf.pkl           ← Trained Random Forest model
│   └── scaler.pkl                  ← Fitted StandardScaler
│
├── data/
│   └── Data_for_UCI_named.csv      ← Raw dataset
│
├── assets/
│   └── screenshot.png              ← App screenshot
│
└── README.md                       ← This file
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Rows** | 10,000 simulated grid readings |
| **Columns** | 14 (12 features + 2 targets) |
| **Target** | `stabf` → `stable` / `unstable` |
| **Class Split** | 63.6% unstable · 36.4% stable |
| **Missing Values** | None |
| **Duplicates** | None |

### Feature Groups

| Group | Columns | Description |
|---|---|---|
| Reaction Time | `tau1` – `tau4` | Reaction times of 4 grid participants (sec) |
| Power | `p1` – `p4` | Power consumed (+) or produced (−) by each node |
| Elasticity | `g1` – `g4` | Price elasticity coefficients |
| Target (numeric) | `stab` | Stability margin (negative = unstable) |
| Target (label) | `stabf` | `stable` / `unstable` |

---

## 🔧 Pipeline — Phase by Phase

```
Phase 1 → Problem Formulation & Data Understanding
Phase 2 → Data Cleaning & Preprocessing
Phase 3 → Feature Engineering & EDA
Phase 4 → Model Building (5 Models + Cross-Validation)
Phase 5 → Model Evaluation & Hyperparameter Tuning
Phase 6 → Deployment (Streamlit App + joblib)
```

### Phase 3 — Feature Engineering
7 new features were created from the raw 12:

| Engineered Feature | Formula |
|---|---|
| `tau_mean` | Mean of tau1–tau4 |
| `tau_std` | Std dev of tau1–tau4 |
| `tau_range` | max(tau) − min(tau) |
| `p_total` | Sum of p1–p4 |
| `p_range` | max(p) − min(p) |
| `g_mean` | Mean of g1–g4 |
| `g_std` | Std dev of g1–g4 |

### Phase 4 — Models Trained

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 0.7810 |
| K-Nearest Neighbors | 0.8720 |
| Decision Tree | 0.8940 |
| Gradient Boosting | 0.9280 |
| **Random Forest ⭐** | **0.9420** |

---

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | 94.20% |
| **Precision** | 95.10% |
| **Recall** | 95.30% |
| **F1 Score** | 95.21% |
| **ROC-AUC** | 0.9820 |

### Best Hyperparameters (GridSearchCV)

```python
{
    'n_estimators'     : 200,
    'max_depth'        : None,
    'max_features'     : 'sqrt',
    'min_samples_split': 2
}
```

### Top 5 Most Important Features

```
tau_mean    →  18.2%
p_total     →  15.9%
tau1        →  14.3%
g_mean      →  12.8%
tau_range   →   9.7%
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-grid-stability.git
cd smart-grid-stability
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Model Files
Open and run the Jupyter notebook end-to-end:
```bash
jupyter notebook notebooks/Capstone_Day50_58_UCI_GridStability.ipynb
```
This generates `best_model_rf.pkl` and `scaler.pkl` in the root folder.

### 5. Run the Streamlit App
```bash
streamlit run app.py
```
Opens at → **http://localhost:8501**

---

## 🌐 Deploy on Streamlit Cloud

1. Push your project to GitHub (include `.pkl` files or generate them in the app)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select your repository
4. Set **Main file path** → `app.py`
5. Set **Python version** → `3.11`
6. Click **Deploy**

> ⚠️ Make sure `requirements.txt` is in the root of your repository before deploying.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11.9 | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| matplotlib + seaborn | Data visualization |
| scikit-learn | ML models, preprocessing, evaluation |
| joblib | Model serialization |
| Streamlit | Web application |
| Jupyter Notebook | Analysis & experimentation |

---

## 📓 Notebook Structure

```
Cell 1  → Install & Import all libraries
Cell 2  → Load dataset
Cell 3  → Data types, nulls, duplicates
Cell 4  → Statistical summary
Cell 5  → Target class distribution
Cell 6  → Outlier detection (IQR)
Cell 7  → Feature distributions (histograms)
Cell 8  → Encode target & feature engineering
Cell 9  → Correlation heatmap
Cell 10 → Boxplots by class
Cell 11 → Train / test split + scaling
Cell 12 → Define 5 models
Cell 13 → 5-Fold cross-validation
Cell 14 → Train all models
Cell 15 → Evaluate — full metrics table
Cell 16 → Confusion matrices
Cell 17 → ROC curves
Cell 18 → GridSearchCV tuning
Cell 19 → Feature importance
Cell 20 → PCA 2D visualization
Cell 21 → Save model with joblib
Cell 22 → Predict on new sample
Cell 23 → Final summary
```

---

## 🔮 How to Use the App

1. Open the app at `http://localhost:8501`
2. Go to the **🔮 Predict** tab
3. Enter the 12 input values:
   - `tau1` – `tau4` : reaction times (0 – 15 sec)
   - `p1` – `p4` : power values (−5 to +10)
   - `g1` – `g4` : elasticity coefficients (0 – 2)
4. The 7 engineered features are **calculated automatically**
5. Click **⚡ Predict Grid Stability**
6. See: **✅ STABLE** or **⚠️ UNSTABLE** with confidence probabilities

---

## 🔭 Future Improvements

- [ ] Connect to a live grid sensor API for real-time predictions
- [ ] Try XGBoost / LightGBM for potentially higher accuracy
- [ ] Add SHAP explainability charts
- [ ] Build a REST API with FastAPI
- [ ] Add user authentication for multi-user deployment
- [ ] Handle class imbalance with SMOTE

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Your Name**  
Data Science Capstone · Day 50–58  
📧 aditikarn167@gmail.com 
🔗 [LinkedIn](https://www.linkedin.com/in/aditikarn-data-analyst/) · [GitHub](https://github.com/aditikarn-analyst)

---

> ⭐ If you found this project helpful, please give it a star on GitHub!