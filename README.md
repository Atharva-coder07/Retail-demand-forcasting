# Retail Demand Forecasting System

An end-to-end Machine Learning system for predicting retail store-SKU panel demand, featuring a leakage-safe feature engineering pipeline, Time-Series Cross Validation (LightGBM), SHAP explainability, foundation-model benchmarking (Chronos/TimeGPT), and an interactive What-If scenario simulator styled with the **Linear dark mode design system**.

---

## Project Structure

```
retail_demand_forecasting/
├── data/
│   └── retail_sales.csv           # Generated synthetic panel dataset (16,440 records)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── generate_data.py           # Phase 1: Multi-series panel data generator
│   ├── features.py                # Phase 2: Leakage-safe feature pipeline
│   ├── models.py                  # Phase 3: LightGBM training and baseline models
│   ├── xai.py                     # Phase 4: SHAP value attribution and plots
│   └── foundation_models.py       # Phase 5: Amazon Chronos & Nixtla TimeGPT wrappers
├── app/
│   └── streamlit_app.py           # Phase 6: Streamlit What-If simulator application
├── outputs/
│   ├── shap_summary.png           # Global beeswarm feature importance plot
│   └── shap_waterfall_example.png # Local waterfall contribution plot
├── requirements.txt               # Package dependencies
└── README.md                      # Phase 7: System report and documentation
```

---

## Setup & Execution

### 1. Clone & Initialize Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Phased Execution in Order
```bash
# Phase 1: Generate synthetic panel dataset
python src/generate_data.py

# Phase 2: Verify feature engineering pipeline
python src/features.py

# Phase 3: Evaluate baseline models vs LightGBM CV
python src/models.py

# Phase 4: Extract SHAP values and save plots
python src/xai.py

# Phase 5: Run zero-shot foundation model benchmark
python src/foundation_models.py

# Phase 6: Start Streamlit interactive dashboard
streamlit run app/streamlit_app.py
```

---

## Model Evaluation & Performance

The forecasting models were evaluated on the synthetic panel dataset using **Mean Absolute Percentage Error (MAPE)**:

| Model / Baseline | Validation Method | MAPE | Best Use Case / Status |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | $y_t = y_{t-1}$ | **32.07%** | Simplest day-lag baseline |
| **Seasonal Naive Baseline** | $y_t = y_{t-7}$ | **28.52%** | Captures weekly seasonality |
| **LightGBM (Warm Start)** | 5-Fold Time-Series CV | **12.79%** | Production model with local context |
| **Amazon Chronos (Mock/Zero-shot)** | Zero-shot Holdout | **15.90%** | Outstanding for cold-start (new SKUs) |
| **Nixtla TimeGPT (Mock/Zero-shot)**| Zero-shot Holdout | **10.94%** | Fast zero-shot forecasting |

> [!NOTE]
> LightGBM significantly outperforms the Seasonal Naive baseline (12.79% vs 28.52%), proving it has successfully learned non-linear price elasticity, promotional boosts, and holiday relationships.

---

## Key Technical Implementations

### 1. Leakage-Safety (Phase 2)
To prevent target leakage (where future info is present in training or today's sales are included in today's features):
- **Lag Features** (`lag_1`, `lag_7`, `lag_14`, `lag_30`) are computed explicitly within `groupby(['store_id', 'sku_id'])` to prevent values from one store-SKU sequence leaking into the first rows of another.
- **Rolling Window Features** (`rolling_mean_7`, `rolling_std_7`, etc.) are computed on a target **shifted by 1 day first** within each store-SKU group before rolling. Rolling on unshifted targets is a common source of inflated offline metrics that fails in production.

### 2. SHAP Explainability (Phase 4)
Using `TreeExplainer`, we decompose predictions into:
- **Base (average) forecast**: ~120 units.
- **Exogenous Drivers**: Toggling promotion (`is_promo = 1`) boosts sales immediately, while price increases (`price = 23.99`) act as a strong negative force due to elasticity.
- **Global Feature Importance**: Saved as a beeswarm plot in `outputs/shap_summary.png` to answer "which features matter most overall."
- **Waterfall Plot**: Decomposes specific predictions in `outputs/shap_waterfall_example.png`.

### 3. "What-If" Simulator with Linear Aesthetics (Phase 6)
The simulator is custom-themed to match the premium, near-black **Linear App** visual identity:
- **Canvas background**: `#010102` (deep, luxurious dark surface).
- **Secondary surfaces (Surface-1)**: `#0f1011` cards with a `1px` solid hairline border (`#23252a`) and `12px` rounded corners.
- **Accent brand color**: Lavender-blue (`#5e6ad2`) focus accents and button states.
- **Inter** display typography with high readability.

---

## Suggested Narrative

1. **Data**: We generate a multi-series panel dataset mapping real-world behaviors (promotions, price changes, weather, and calendar events).
2. **Baseline**: We establish simple benchmarks (Naive and Seasonal Naive) to set a floor on forecasting capability.
3. **Model**: We train a gradient-boosted decision tree (LightGBM) using chronological time-series splitting to prove it learns the relationships and outperforms baselines.
4. **Explainability**: We unlock the black box by visualizing global feature attributions and row-level waterfalls using SHAP.
5. **Frontier Check**: We frame LightGBM against zero-shot foundation models (Chronos/TimeGPT) to understand the cold-start vs. warm-start trade-off.
6. **Strategy Tool**: We deliver an interactive decision-support application where commercial teams can simulate price changes, promotion plans, or weather scenarios and instantly view explainable outcomes.
