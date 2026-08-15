import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Add the project root to the path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.features import build_feature_frame, get_feature_columns
from src.models import train_final_lgbm
from src.xai import compute_shap_values, narrate_shap_explanation, format_contributions
from src.anomaly_report import detect_anomalies, narrate_anomaly, generate_daily_report

# Set page config
st.set_page_config(
    page_title="Linear Demand Forecasting Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom Linear design system styling via CSS injection
st.markdown(
    """
    <style>
    /* Primary canvas background */
    .stApp {
        background-color: #010102;
        color: #f7f8f8;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f7f8f8 !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        letter-spacing: -0.05em;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1011 !important;
        border-right: 1px solid #23252a !important;
    }
    
    /* Sidebar content text color override */
    [data-testid="stSidebar"] * {
        color: #d0d6e0 !important;
    }
    
    /* Custom metric container styling */
    .metric-container {
        background-color: #0f1011;
        border: 1px solid #23252a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: none;
    }
    
    .metric-title {
        color: #8a8f98;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #f7f8f8;
        font-size: 36px;
        font-weight: 600;
        letter-spacing: -0.03em;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .delta-positive {
        color: #27a644;
    }
    
    .delta-negative {
        color: #f85149;
    }
    
    /* Dividers and borders */
    hr {
        border-color: #23252a !important;
    }
    
    /* Style inputs */
    input, select, .stSlider {
        background-color: #141516 !important;
        color: #f7f8f8 !important;
    }
    
    /* Info box styling */
    div.stAlert {
        background-color: #0f1011 !important;
        color: #d0d6e0 !important;
        border: 1px solid #23252a !important;
        border-radius: 8px !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #010102;
        border-bottom: 1px solid #23252a;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8a8f98 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f7f8f8 !important;
        border-bottom-color: #5e6ad2 !important;
    }
    
    /* AI interpretation box */
    .ai-box {
        background: linear-gradient(135deg, #0f1011 0%, #131420 100%);
        border: 1px solid #2d2f6d;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        position: relative;
    }
    .ai-box::before {
        content: "✦ AI Interpretation";
        display: block;
        color: #5e6ad2;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .ai-box p {
        color: #c8cdd5;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    .ai-disclaimer {
        color: #555a63;
        font-size: 11px;
        margin-top: 8px;
        font-style: italic;
    }
    
    /* Chat styling */
    .stChatMessage {
        background-color: #0f1011 !important;
        border: 1px solid #23252a !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ─── Gemini scenario advice (Phase 8.2) ──────────────────────────────────────

def generate_scenario_advice(baseline_pred, scenario_pred, scenario_params, unit_price):
    """
    Computes revenue deltas in Python, then asks Gemini for a 2-3 sentence
    strategic recommendation. Falls back to a template on API error.
    """
    unit_delta_pct = (scenario_pred - baseline_pred) / max(baseline_pred, 1e-6) * 100
    baseline_revenue = baseline_pred * unit_price
    scenario_revenue = scenario_pred * (unit_price * (1 + scenario_params['price_change_pct'] / 100))
    revenue_delta_pct = (scenario_revenue - baseline_revenue) / max(baseline_revenue, 1e-6) * 100

    try:
        from llm import get_model
        
        prompt = f"""You are a retail pricing/planning advisor. Given the scenario below, write a
2-3 sentence strategic recommendation. Use ONLY the numbers provided — do not
invent additional statistics. It is fine to add ONE general, clearly-labeled
caveat (e.g. "this doesn't account for competitor response") but do not
present that caveat as a computed fact.

Scenario applied: price change {scenario_params['price_change_pct']:+.0f}%, promo={scenario_params['run_promo']}, simulated holiday={scenario_params['force_holiday']}
Baseline forecast: {baseline_pred:.0f} units (${baseline_revenue:,.0f} revenue)
Scenario forecast: {scenario_pred:.0f} units ({unit_delta_pct:+.1f}%), (${scenario_revenue:,.0f} revenue, {revenue_delta_pct:+.1f}%)

Write the recommendation now."""
        
        response = get_model().generate_content(prompt)
        return response.text.strip()
    except Exception:
        return (
            f"Scenario changes forecast by {unit_delta_pct:+.1f}% in units "
            f"and {revenue_delta_pct:+.1f}% in revenue "
            f"(${baseline_revenue:,.0f} → ${scenario_revenue:,.0f})."
        )


# ─── NL Q&A intent parsing (Phase 8.4) ───────────────────────────────────────

def parse_user_intent(user_question, store_ids, sku_ids):
    """
    Step 1: Ask Gemini to convert the user's question into a constrained JSON object.
    Returns a dict with keys: intent, store_id, sku_id, time_window_days.
    """
    try:
        from llm import get_model
        
        prompt = f"""Convert the user's question into JSON with this exact schema. Only use
values that are valid given the lists provided — never invent a store_id or
sku_id that isn't in the list.

Valid store_ids: {list(store_ids)}
Valid sku_ids: {list(sku_ids)}

Schema:
{{
  "intent": "compare_growth" | "single_forecast" | "explain_anomaly" | "unknown",
  "store_id": <one of valid store_ids or null>,
  "sku_id": <one of valid sku_ids or null>,
  "time_window_days": <int or null>
}}

User question: "{user_question}"

Return ONLY the JSON, no other text."""
        
        response = get_model().generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return json.loads(raw)
    except Exception:
        return {"intent": "unknown", "store_id": None, "sku_id": None, "time_window_days": None}


def execute_intent(intent_data, df_feat, features, model):
    """
    Step 2: Deterministic backend computation based on parsed intent.
    Returns a dict with the computed result.
    """
    intent = intent_data.get("intent", "unknown")
    store_id = intent_data.get("store_id")
    sku_id = intent_data.get("sku_id")
    window = intent_data.get("time_window_days", 7) or 7
    
    if intent == "unknown":
        return {"type": "unknown", "message": "I couldn't map that to a supported question. Try asking about a specific store/SKU's forecast, growth comparison, or anomaly explanation."}
    
    # Filter data
    mask = pd.Series(True, index=df_feat.index)
    if store_id:
        mask &= df_feat['store_id'] == store_id
    if sku_id:
        mask &= df_feat['sku_id'] == sku_id
    
    subset = df_feat[mask].copy()
    if subset.empty:
        return {"type": "error", "message": f"No data found for store={store_id}, SKU={sku_id}."}
    
    subset = subset.sort_values('date')
    
    if intent == "single_forecast":
        # Get the latest row and predict
        latest = subset.iloc[-1:]
        feat_cols = [c for c in features if c in latest.columns]
        pred_input = latest[feat_cols].copy()
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in pred_input.columns:
                pred_input[col] = pred_input[col].astype('category')
        pred = max(0, float(model.predict(pred_input)[0]))
        return {
            "type": "single_forecast",
            "store_id": store_id or "all",
            "sku_id": sku_id or "all",
            "date": str(latest['date'].iloc[0]),
            "forecast": pred,
        }
    
    elif intent == "compare_growth":
        # Compute recent vs prior window forecasts
        recent = subset.tail(window)
        prior = subset.iloc[-(window*2):-window] if len(subset) >= window * 2 else subset.head(window)
        
        feat_cols = [c for c in features if c in recent.columns]
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in recent.columns:
                recent[col] = recent[col].astype('category')
                prior[col] = prior[col].astype('category')
        
        recent_avg = max(0, float(np.mean(model.predict(recent[feat_cols]))))
        prior_avg = max(0, float(np.mean(model.predict(prior[feat_cols]))))
        growth_pct = ((recent_avg - prior_avg) / max(prior_avg, 1e-6)) * 100
        
        return {
            "type": "compare_growth",
            "store_id": store_id or "all",
            "sku_id": sku_id or "all",
            "window_days": window,
            "recent_avg_forecast": recent_avg,
            "prior_avg_forecast": prior_avg,
            "growth_pct": growth_pct,
        }
    
    elif intent == "explain_anomaly":
        # Find biggest anomaly in recent window
        recent = subset.tail(window)
        feat_cols = [c for c in features if c in recent.columns]
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in recent.columns:
                recent[col] = recent[col].astype('category')
        preds = np.clip(model.predict(recent[feat_cols]), 0, None)
        
        predictions_df = recent[['date', 'store_id', 'sku_id']].copy()
        predictions_df['prediction'] = preds
        
        anomalies = detect_anomalies(df_feat, predictions_df, z_threshold=1.0)
        if anomalies.empty:
            return {"type": "no_anomaly", "message": f"No significant forecast anomalies detected in the last {window} days for this panel."}
        
        top_anomaly = anomalies.iloc[0]
        explanation = narrate_anomaly(top_anomaly)
        return {
            "type": "explain_anomaly",
            "date": str(top_anomaly['date']),
            "store_id": top_anomaly['store_id'],
            "sku_id": top_anomaly['sku_id'],
            "prediction": float(top_anomaly['prediction']),
            "rolling_mean": float(top_anomaly['rolling_mean_30']),
            "z_score": float(top_anomaly['z_score']),
            "explanation": explanation,
        }
    
    return {"type": "unknown", "message": "Could not process that intent."}


def narrate_result(result_data):
    """
    Step 3: Optionally narrate the computed result via Gemini, or template directly.
    """
    rtype = result_data.get("type")
    
    if rtype in ("unknown", "error", "no_anomaly"):
        return result_data.get("message", "I couldn't process that question.")
    
    if rtype == "single_forecast":
        return f"The latest forecast for **{result_data['sku_id']}** at **{result_data['store_id']}** (date: {result_data['date']}) is **{result_data['forecast']:.0f} units**."
    
    if rtype == "compare_growth":
        direction = "up" if result_data['growth_pct'] > 0 else "down"
        return (
            f"Over the last **{result_data['window_days']} days**, the average forecast for "
            f"**{result_data['sku_id']}** at **{result_data['store_id']}** is "
            f"**{result_data['recent_avg_forecast']:.0f} units/day** — "
            f"that's **{abs(result_data['growth_pct']):.1f}% {direction}** from the prior "
            f"{result_data['window_days']}-day average of {result_data['prior_avg_forecast']:.0f} units/day."
        )
    
    if rtype == "explain_anomaly":
        return (
            f"**Anomaly on {result_data['date']}** ({result_data['store_id']}/{result_data['sku_id']}): "
            f"forecast was {result_data['prediction']:.0f} units vs 30-day avg of "
            f"{result_data['rolling_mean']:.0f} (z-score: {result_data['z_score']:+.1f}).  \n"
            f"_{result_data['explanation']}_"
        )
    
    return "Result computed but no narration template matched."


# ─── Cached data loading ─────────────────────────────────────────────────────

@st.cache_data
def load_historical_data():
    """
    Loads synthetic dataset.
    """
    data_path = "data/retail_sales.csv"
    if not os.path.exists(data_path):
        # Fallback to generate data if not present
        from src.generate_data import generate_retail_data
        df = generate_retail_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df

@st.cache_resource
def train_and_cache_model(df_feat, features):
    """
    Trains the final LightGBM model and SHAP explainer once per session.
    """
    model, df_clean = train_final_lgbm(df_feat, features)
    
    # Pre-compute SHAP explainer on a background sample to optimize load time
    np.random.seed(42)
    sample_indices = np.random.choice(df_clean.index, size=min(300, len(df_clean)), replace=False)
    X_sample = df_clean.loc[sample_indices, features]
    
    # Cast variables to category
    categorical_features = ['store_id', 'sku_id', 'temp_band']
    for col in categorical_features:
        X_sample[col] = X_sample[col].astype('category')
        
    explainer = shap.TreeExplainer(model)
    
    # Store category metadata
    categories_dict = {
        col: df_clean[col].astype('category').cat.categories for col in categorical_features
    }
    
    return model, explainer, X_sample, categories_dict, df_clean


def main():
    # Header block
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px;">
            <div style="width: 24px; height: 24px; background-color: #5e6ad2; border-radius: 6px;"></div>
            <div style="font-size: 24px; font-weight: 600; letter-spacing: -0.05em; color: #f7f8f8;">
                Linear Demand Forecasting Simulator
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load raw and feature engineered data
    df_raw = load_historical_data()
    
    # Cache feature building
    with st.spinner("Engineering features and compiling model..."):
        df_feat = build_feature_frame(df_raw)
        features_info = get_feature_columns()
        features = features_info['all']
        model, explainer, X_sample, categories_dict, df_clean = train_and_cache_model(df_feat, features)

    # Sidebar parameters
    st.sidebar.markdown("### Select Product Panel")
    store_selected = st.sidebar.selectbox("Store ID", sorted(df_raw['store_id'].unique()))
    sku_selected = st.sidebar.selectbox("SKU ID", sorted(df_raw['sku_id'].unique()))
    
    # Fetch historical series for selected store/SKU
    series_df = df_feat[(df_feat['store_id'] == store_selected) & (df_feat['sku_id'] == sku_selected)].copy()
    series_df = series_df.sort_values('date').reset_index(drop=True)
    
    # Get the latest row for "What-If" simulation
    latest_row = series_df.iloc[-1:].copy()
    latest_date = latest_row['date'].iloc[0] if isinstance(latest_row['date'].iloc[0], str) else str(latest_row['date'].iloc[0])
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(f"### What-If Parameters (Simulating {latest_date})")
    
    # Controls for overrides
    price_pct = st.sidebar.slider("Price Change (%)", -30.0, 30.0, 0.0, step=1.0)
    base_price = float(latest_row['price'].iloc[0])
    sim_price = base_price * (1.0 + price_pct / 100.0)
    st.sidebar.markdown(f"Simulated Price: **${sim_price:.2f}** (Base: ${base_price:.2f})")
    
    sim_promo = st.sidebar.toggle("Active Promotion", value=bool(latest_row['promo_flag'].iloc[0]))
    sim_holiday = st.sidebar.toggle("Holiday Mode", value=bool(latest_row['is_holiday'].iloc[0]))
    
    temp_override = st.sidebar.toggle("Override Temperature", value=False)
    if temp_override:
        sim_temp = st.sidebar.slider("Temperature (°C)", -5.0, 40.0, float(latest_row['temperature'].iloc[0]))
    else:
        sim_temp = float(latest_row['temperature'].iloc[0])
        
    sim_precip = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, float(latest_row['precipitation'].iloc[0]))

    # Build simulation inputs
    baseline_input = latest_row[features].copy()
    scenario_input = latest_row[features].copy()
    
    # Apply modifications to scenario
    scenario_input['price'] = sim_price
    scenario_input['is_promo'] = int(sim_promo)
    scenario_input['is_holiday'] = int(sim_holiday)
    scenario_input['temperature'] = sim_temp
    scenario_input['precipitation'] = sim_precip
    scenario_input['is_rainy'] = int(sim_precip > 5.0)
    
    # Bin temperature band
    temp_bins = [-np.inf, 5, 15, 25, np.inf]
    temp_labels = ['Cold', 'Cool', 'Mild', 'Hot']
    sim_band = pd.cut([sim_temp], bins=temp_bins, labels=temp_labels)[0]
    scenario_input['temp_band'] = sim_band

    # Ensure categorical dtypes match training
    for col in ['store_id', 'sku_id', 'temp_band']:
        baseline_input[col] = pd.Categorical([baseline_input[col].iloc[0]], categories=categories_dict[col])
        scenario_input[col] = pd.Categorical([scenario_input[col].iloc[0]], categories=categories_dict[col])

    # Run predictions
    pred_base = max(0.0, float(model.predict(baseline_input)[0]))
    pred_scen = max(0.0, float(model.predict(scenario_input)[0]))
    
    pct_delta = ((pred_scen - pred_base) / pred_base) * 100 if pred_base > 0 else 0.0

    # Layout: Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">BASELINE FORECAST</div>
                <div class="metric-value">{pred_base:.1f} <span style="font-size:16px; color:#8a8f98;">units</span></div>
                <div class="metric-delta" style="color:#8a8f98;">No overrides applied</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        delta_class = "delta-positive" if pct_delta >= 0 else "delta-negative"
        sign = "+" if pct_delta >= 0 else ""
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">SIMULATED FORECAST</div>
                <div class="metric-value">{pred_scen:.1f} <span style="font-size:16px; color:#8a8f98;">units</span></div>
                <div class="metric-delta {delta_class}">{sign}{pct_delta:.2f}% shift vs baseline</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        price_diff = sim_price - base_price
        price_sign = "+" if price_diff >= 0 else ""
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">SIMULATED PRICE</div>
                <div class="metric-value">${sim_price:.2f}</div>
                <div class="metric-delta" style="color:#d0d6e0;">{price_sign}${price_diff:.2f} ({price_sign}{price_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ─── Phase 8.2: Strategic Advice Box ──────────────────────────────
    scenario_params = {
        'price_change_pct': price_pct,
        'run_promo': sim_promo,
        'force_holiday': sim_holiday,
    }
    advice = generate_scenario_advice(pred_base, pred_scen, scenario_params, base_price)
    st.markdown(
        f"""
        <div class="ai-box">
            <p>{advice}</p>
            <div class="ai-disclaimer">AI-generated interpretation — not a guarantee. Based on pre-computed model outputs.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tabs for visualization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Explainable AI (SHAP)", "Historical Trends", "Model Details", 
        "📊 Anomaly Report", "💬 Ask the Forecast"
    ])
    
    with tab1:
        st.markdown("### Why did the forecast change? (Local SHAP Attribution)")
        st.write("The waterfall plot below decomposes the simulated prediction from the average baseline model forecast.")
        
        # Calculate local SHAP values for the scenario
        shap_values_scen = explainer.shap_values(scenario_input)
        if isinstance(shap_values_scen, list):
            shap_values_scen = shap_values_scen[0]
            
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]
            
        explanation = shap.Explanation(
            values=shap_values_scen[0],
            base_values=base_val,
            data=scenario_input.iloc[0].values,
            feature_names=list(features)
        )
        
        # Plot SHAP waterfall
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1011')
        ax.set_facecolor('#0f1011')
        
        shap.plots.waterfall(explanation, max_display=7, show=False)
        
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#f7f8f8')
        ax.title.set_color('#f7f8f8')
        
        st.pyplot(fig)
        plt.close()
        
        # ─── Phase 8.1: SHAP Narration ───────────────────────────────
        # Build top_contributions dict from SHAP values
        shap_vals = shap_values_scen[0]
        feat_names = list(features)
        abs_vals = np.abs(shap_vals)
        top_indices = np.argsort(abs_vals)[-5:][::-1]
        top_contributions = {feat_names[i]: float(shap_vals[i]) for i in top_indices}
        
        predicted_value = base_val + float(np.sum(shap_vals))
        context = {
            'store_id': store_selected,
            'sku_id': sku_selected,
            'date': latest_date,
        }
        
        narration = narrate_shap_explanation(base_val, predicted_value, top_contributions, context)
        
        st.markdown(
            f"""
            <div class="ai-box">
                <p>{narration}</p>
                <div class="ai-disclaimer">Narration generated from pre-computed SHAP values — not independent analysis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with tab2:
        st.markdown("### Historical Sales & Pricing Trend")
        
        hist_subset = series_df.iloc[-60:].copy()
        
        fig, ax1 = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#0f1011')
        ax1.set_facecolor('#0f1011')
        
        color = '#5e6ad2'
        ax1.set_xlabel('Date', color='#8a8f98')
        ax1.set_ylabel('Sales (units)', color=color)
        ax1.plot(pd.to_datetime(hist_subset['date']), hist_subset['sales'], color=color, linewidth=2, label='Actual Sales')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(colors='#8a8f98')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#23252a')
        ax1.spines['bottom'].set_color('#23252a')
        
        ax2 = ax1.twinx()
        color_price = '#27a644'
        ax2.set_ylabel('Price ($)', color=color_price)
        ax2.step(pd.to_datetime(hist_subset['date']), hist_subset['price'], color=color_price, where='post', alpha=0.7, label='Price')
        ax2.tick_params(axis='y', labelcolor=color_price)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('#23252a')
        ax2.spines['bottom'].set_color('#23252a')
        
        plt.title(f"Last 60 Days: {sku_selected} at {store_selected}", color='#f7f8f8', pad=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with tab3:
        st.markdown("### Model Properties & Exogenous Context")
        
        st.markdown("#### Latest Features State (Baseline vs Simulated)")
        comp_df = pd.DataFrame({
            'Feature': features,
            'Baseline Value': baseline_input.iloc[0].values,
            'Simulated Value': scenario_input.iloc[0].values
        })
        important_features = ['price', 'is_promo', 'is_holiday', 'temperature', 'precipitation', 'is_rainy', 'temp_band', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30']
        comp_df = comp_df[comp_df['Feature'].isin(important_features)]
        st.table(comp_df)

    # ─── Phase 8.3: Anomaly Report Tab ────────────────────────────────
    with tab4:
        st.markdown("### Forecast Anomaly Detection")
        st.write("Flags days where the forecast deviates significantly from the 30-day rolling average, then uses AI to explain likely drivers.")
        
        z_thresh = st.slider("Z-Score Threshold", 1.0, 3.0, 1.5, step=0.1, key="anomaly_z")
        
        # Generate predictions for the selected panel
        panel_data = series_df.copy()
        panel_features = [c for c in features if c in panel_data.columns]
        panel_pred_input = panel_data[panel_features].copy()
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in panel_pred_input.columns:
                panel_pred_input[col] = panel_pred_input[col].astype('category')
        
        panel_preds = np.clip(model.predict(panel_pred_input), 0, None)
        predictions_df = panel_data[['date', 'store_id', 'sku_id']].copy()
        predictions_df['prediction'] = panel_preds
        
        anomalies = detect_anomalies(df_feat, predictions_df, z_threshold=z_thresh)
        
        if anomalies.empty:
            st.info(f"No anomalies detected above z={z_thresh:.1f} for {store_selected}/{sku_selected}.")
        else:
            st.markdown(f"**{len(anomalies)} anomalies detected** (z ≥ {z_thresh:.1f})")
            
            # Show top 8
            for _, row in anomalies.head(8).iterrows():
                direction_icon = "🔺" if row['z_score'] > 0 else "🔻"
                explanation = narrate_anomaly(row)
                
                st.markdown(
                    f"""
                    <div class="ai-box">
                        <p><strong>{direction_icon} {row['date']}</strong> — Forecast: {row['prediction']:.0f} units vs avg {row['rolling_mean_30']:.0f} (z={row['z_score']:+.1f})</p>
                        <p>{explanation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ─── Phase 8.4: NL Q&A Tab ────────────────────────────────────────
    with tab5:
        st.markdown("### Ask a Question About the Forecast")
        st.write("Ask in plain English. The system parses your intent, computes the answer from real data, then phrases it naturally.")
        
        st.markdown(
            """
            <div style="color: #8a8f98; font-size: 13px; margin-bottom: 12px;">
            <strong>Try:</strong> "Which SKU has the highest forecast next week at S1?" · 
            "How is growth trending for SKU003?" · 
            "Explain any anomalies for S2/SKU001"
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        user_question = st.chat_input("Ask about forecasts, growth, or anomalies...")
        
        if user_question:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    store_ids = sorted(df_raw['store_id'].unique())
                    sku_ids = sorted(df_raw['sku_id'].unique())
                    
                    # Step 1: Parse intent
                    intent_data = parse_user_intent(user_question, store_ids, sku_ids)
                    
                    # Step 2: Deterministic computation
                    result = execute_intent(intent_data, df_feat, features, model)
                    
                    # Step 3: Narrate result
                    answer = narrate_result(result)
                    
                st.markdown(answer)
                st.caption("_AI-assisted answer — computed from real model data, not generated from scratch._")
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
