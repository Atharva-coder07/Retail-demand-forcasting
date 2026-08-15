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
    page_title="RetailIntel Pro — Demand Forecasting Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom Linear / Obsidian Precision design system styling via CSS injection
st.markdown(
    """
    <style>
    /* Primary canvas background */
    .stApp {
        background-color: #010102;
        color: #f7f8f8;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Top Header Navbar */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background-color: #0e0e11;
        border-bottom: 1px solid #23252a;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    .brand-logo {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #f7f8f8;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-links {
        display: flex;
        gap: 24px;
        font-size: 14px;
        font-weight: 500;
        color: #8a8f98;
    }
    .nav-links span:hover {
        color: #f7f8f8;
        cursor: pointer;
    }
    .nav-pill-badge {
        background-color: #1a1b24;
        color: #bdc2ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #2e3aa2;
    }
    .export-btn {
        background-color: #5e6ad2;
        color: #ffffff;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 6px;
        border: none;
        cursor: pointer;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1011 !important;
        border-right: 1px solid #23252a !important;
    }
    
    /* Sidebar nav links */
    .sidebar-section-title {
        color: #8a8f98;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        color: #8a8f98;
        font-size: 14px;
        border-radius: 6px;
        margin-bottom: 4px;
    }
    .sidebar-nav-item.active {
        background-color: #1a1b24;
        color: #f7f8f8;
        font-weight: 500;
        border-left: 3px solid #5e6ad2;
    }

    /* Custom metric container styling */
    .metric-card {
        background-color: #0f1011;
        border: 1px solid #23252a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        position: relative;
    }
    .metric-card.active-highlight {
        border: 1px solid #5e6ad2;
        box-shadow: 0 0 15px rgba(94, 106, 210, 0.15);
    }
    
    .metric-label {
        color: #8a8f98;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-val {
        color: #f7f8f8;
        font-size: 32px;
        font-weight: 600;
        letter-spacing: -0.03em;
    }
    
    .pill-green {
        display: inline-block;
        background-color: rgba(39, 166, 68, 0.15);
        color: #27a644;
        border: 1px solid rgba(39, 166, 68, 0.3);
        font-size: 13px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-top: 8px;
    }
    .pill-red {
        display: inline-block;
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
        font-size: 13px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-top: 8px;
    }

    /* AI Strategic Recommendation Banner */
    .ai-recommendation-card {
        background: linear-gradient(135deg, #0f1011 0%, #161726 100%);
        border: 1px solid #303478;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .ai-rec-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #bdc2ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .ai-star-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background-color: #2e3aa2;
        border-radius: 50%;
        color: #bdc2ff;
        font-size: 14px;
    }
    .ai-rec-body {
        color: #e5e1e6;
        font-size: 15px;
        line-height: 1.6;
    }
    .text-green-highlight {
        color: #27a644;
        font-weight: 600;
    }

    /* Copilot Panel Styling */
    .copilot-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 12px;
        border-bottom: 1px solid #23252a;
        margin-bottom: 16px;
    }
    .copilot-title {
        font-size: 16px;
        font-weight: 600;
        color: #f7f8f8;
        display: flex;
        align-items: center;
        gap: 8px;
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
    revenue_delta = scenario_revenue - baseline_revenue

    try:
        from llm import get_model
        
        prompt = f"""You are a retail pricing/planning advisor. Given the scenario below, write a
2-3 sentence strategic recommendation. Use ONLY the numbers provided — do not
invent additional statistics. It is fine to add ONE general, clearly-labeled
caveat (e.g. "this doesn't account for competitor response") but do not
present that caveat as a computed fact.

Scenario applied: price change {scenario_params['price_change_pct']:+.0f}%, promo={scenario_params['run_promo']}, simulated holiday={scenario_params['force_holiday']}
Baseline forecast: {baseline_pred:.0f} units (${baseline_revenue:,.0f} revenue)
Scenario forecast: {scenario_pred:.0f} units ({unit_delta_pct:+.1f}%), (${scenario_revenue:,.0f} revenue, net change ${revenue_delta:+,.0f})

Write the recommendation now."""
        
        response = get_model().generate_content(prompt)
        return response.text.strip()
    except Exception:
        sign_str = "+" if revenue_delta >= 0 else ""
        return (
            f"Increasing price by {scenario_params['price_change_pct']:+.0f}% alongside active promo "
            f"generates net {sign_str}${revenue_delta:,.0f} revenue. Elasticity model predicts "
            f"demand will shift by {unit_delta_pct:+.1f}%."
        )


# ─── NL Q&A intent parsing (Phase 8.4) ───────────────────────────────────────

def parse_user_intent(user_question, store_ids, sku_ids, current_store, current_sku):
    try:
        from llm import get_model
        
        prompt = f"""You are a retail forecast assistant. Convert the user's question into JSON.

The user is currently viewing: Store={current_store}, SKU={current_sku}.
If the user doesn't specify a store or SKU, use these defaults.
Only use store_ids and sku_ids from the valid lists — never invent values.

Valid store_ids: {list(store_ids)}
Valid sku_ids: {list(sku_ids)}

Schema:
{{
  "intent": "compare_growth" | "single_forecast" | "explain_anomaly" | "best_sku" | "summary" | "general_chat",
  "store_id": <one of valid store_ids or null>,
  "sku_id": <one of valid sku_ids or null>,
  "time_window_days": <int or null>
}}

User question: "{user_question}"

Return ONLY the JSON, no other text."""
        
        response = get_model().generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        parsed = json.loads(raw)
        if not parsed.get('store_id'):
            parsed['store_id'] = current_store
        if not parsed.get('sku_id') and parsed.get('intent') != 'best_sku':
            parsed['sku_id'] = current_sku
        return parsed
    except Exception:
        return {"intent": "general_chat", "store_id": current_store, "sku_id": current_sku, "time_window_days": None}


def execute_intent(intent_data, df_feat, features, model, all_sku_ids=None):
    intent = intent_data.get("intent", "general_chat")
    store_id = intent_data.get("store_id")
    sku_id = intent_data.get("sku_id")
    window = intent_data.get("time_window_days", 7) or 7
    
    mask = pd.Series(True, index=df_feat.index)
    if store_id:
        mask &= df_feat['store_id'] == store_id
    if sku_id and intent not in ('best_sku',):
        mask &= df_feat['sku_id'] == sku_id
    
    subset = df_feat[mask].copy()
    if subset.empty:
        return {"type": "error", "message": f"No data found for store={store_id}, SKU={sku_id}."}
    
    subset = subset.sort_values('date')
    
    if intent == "single_forecast":
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
        recent = subset.tail(max(window, 30))
        feat_cols = [c for c in features if c in recent.columns]
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in recent.columns:
                recent[col] = recent[col].astype('category')
        preds = np.clip(model.predict(recent[feat_cols]), 0, None)
        
        predictions_df = recent[['date', 'store_id', 'sku_id']].copy()
        predictions_df['prediction'] = preds
        
        anomalies = detect_anomalies(df_feat, predictions_df, z_threshold=1.0)
        if anomalies.empty:
            return {"type": "no_anomaly", "message": f"No significant forecast anomalies detected in the last {window} days for {store_id}/{sku_id}."}
        
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
    
    elif intent == "best_sku":
        store_data = df_feat[df_feat['store_id'] == store_id].copy()
        skus = all_sku_ids or store_data['sku_id'].unique()
        sku_forecasts = []
        for s in skus:
            sku_rows = store_data[store_data['sku_id'] == s].sort_values('date')
            if sku_rows.empty:
                continue
            latest = sku_rows.iloc[-1:]
            feat_cols = [c for c in features if c in latest.columns]
            pred_input = latest[feat_cols].copy()
            for col in ['store_id', 'sku_id', 'temp_band']:
                if col in pred_input.columns:
                    pred_input[col] = pred_input[col].astype('category')
            pred = max(0, float(model.predict(pred_input)[0]))
            sku_forecasts.append({'sku_id': s, 'forecast': pred})
        
        sku_forecasts.sort(key=lambda x: x['forecast'], reverse=True)
        return {"type": "best_sku", "store_id": store_id, "rankings": sku_forecasts}
    
    elif intent == "summary":
        latest = subset.iloc[-1:]
        feat_cols = [c for c in features if c in latest.columns]
        pred_input = latest[feat_cols].copy()
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in pred_input.columns:
                pred_input[col] = pred_input[col].astype('category')
        pred = max(0, float(model.predict(pred_input)[0]))
        
        recent7 = subset.tail(7)
        prior7 = subset.iloc[-14:-7] if len(subset) >= 14 else subset.head(7)
        feat_cols_r = [c for c in features if c in recent7.columns]
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in recent7.columns:
                recent7[col] = recent7[col].astype('category')
                prior7[col] = prior7[col].astype('category')
        recent_avg = max(0, float(np.mean(model.predict(recent7[feat_cols_r]))))
        prior_avg = max(0, float(np.mean(model.predict(prior7[feat_cols_r]))))
        growth = ((recent_avg - prior_avg) / max(prior_avg, 1e-6)) * 100
        
        return {
            "type": "summary",
            "store_id": store_id,
            "sku_id": sku_id,
            "latest_date": str(latest['date'].iloc[0]),
            "latest_forecast": pred,
            "avg_7d": recent_avg,
            "growth_7d_pct": growth,
            "has_promo": bool(latest['is_promo'].iloc[0]) if 'is_promo' in latest.columns else False,
            "has_holiday": bool(latest['is_holiday'].iloc[0]) if 'is_holiday' in latest.columns else False,
        }
    
    elif intent == "general_chat":
        latest = subset.iloc[-1:]
        feat_cols = [c for c in features if c in latest.columns]
        pred_input = latest[feat_cols].copy()
        for col in ['store_id', 'sku_id', 'temp_band']:
            if col in pred_input.columns:
                pred_input[col] = pred_input[col].astype('category')
        pred = max(0, float(model.predict(pred_input)[0]))
        
        return {
            "type": "general_chat",
            "store_id": store_id,
            "sku_id": sku_id,
            "latest_forecast": pred,
            "latest_date": str(latest['date'].iloc[0]),
            "user_question": intent_data.get('_original_question', ''),
        }
    
    return {"type": "general_chat", "store_id": store_id, "sku_id": sku_id, "latest_forecast": 0, "latest_date": "N/A", "user_question": ""}


def narrate_result(result_data, user_question=""):
    rtype = result_data.get("type")
    if rtype in ("error", "no_anomaly"):
        return result_data.get("message", "I couldn't process that question.")
    
    if rtype == "single_forecast":
        return f"📊 The latest forecast for **{result_data['sku_id']}** at **{result_data['store_id']}** (date: {result_data['date']}) is **{result_data['forecast']:.0f} units**."
    
    if rtype == "compare_growth":
        direction = "up" if result_data['growth_pct'] > 0 else "down"
        icon = "📈" if result_data['growth_pct'] > 0 else "📉"
        return (
            f"{icon} Over the last **{result_data['window_days']} days**, the average forecast for "
            f"**{result_data['sku_id']}** at **{result_data['store_id']}** is "
            f"**{result_data['recent_avg_forecast']:.0f} units/day** — "
            f"that's **{abs(result_data['growth_pct']):.1f}% {direction}** from the prior "
            f"{result_data['window_days']}-day average of {result_data['prior_avg_forecast']:.0f} units/day."
        )
    
    if rtype == "explain_anomaly":
        return (
            f"⚠️ **Anomaly on {result_data['date']}** ({result_data['store_id']}/{result_data['sku_id']}): "
            f"forecast was {result_data['prediction']:.0f} units vs 30-day avg of "
            f"{result_data['rolling_mean']:.0f} (z-score: {result_data['z_score']:+.1f}).  \n"
            f"_{result_data['explanation']}_"
        )
    
    if rtype == "best_sku":
        rankings = result_data.get('rankings', [])
        lines = [f"🏆 **SKU Rankings at {result_data['store_id']}** (by latest forecast):"]
        for i, r in enumerate(rankings):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            lines.append(f"  {medal} **{r['sku_id']}** — {r['forecast']:.0f} units/day")
        return "\n".join(lines)
    
    if rtype == "summary":
        d = result_data
        growth_icon = "📈" if d['growth_7d_pct'] > 0 else "📉" if d['growth_7d_pct'] < 0 else "➡️"
        flags = []
        if d.get('has_promo'):
            flags.append("🏷️ Promo active")
        if d.get('has_holiday'):
            flags.append("🎄 Holiday")
        flag_str = " · ".join(flags) if flags else "No active promotions or holidays"
        
        return (
            f"📋 **Summary for {d['sku_id']} at {d['store_id']}** (as of {d['latest_date']}):\n\n"
            f"• Latest forecast: **{d['latest_forecast']:.0f} units**\n"
            f"• 7-day average: **{d['avg_7d']:.0f} units/day**\n"
            f"• {growth_icon} 7-day trend: **{d['growth_7d_pct']:+.1f}%**\n"
            f"• Status: {flag_str}"
        )
    
    if rtype == "general_chat":
        try:
            from llm import get_model
            prompt = f"""You are a helpful retail forecast assistant embedded in a demand forecasting dashboard.
The user is currently viewing data for Store: {result_data.get('store_id')}, SKU: {result_data.get('sku_id')}.
Latest forecast: {result_data.get('latest_forecast', 0):.0f} units (date: {result_data.get('latest_date', 'N/A')}).

The user asked: "{user_question}"

Respond helpfully in 1-3 sentences. If they're greeting you, greet them back and briefly mention
what you can help with (forecasts, growth trends, anomalies, SKU rankings).
If they ask something outside your scope, politely say so and suggest what you CAN answer.
Do not invent data — only reference the numbers provided above."""
            response = get_model().generate_content(prompt)
            return response.text.strip()
        except Exception:
            return (
                f"👋 Hi! I'm your forecast assistant for **{result_data.get('store_id')}/{result_data.get('sku_id')}**. "
                f"The latest forecast is **{result_data.get('latest_forecast', 0):.0f} units**. "
                f"Try asking about **growth trends**, **anomalies**, **best SKU**, or **forecasts**!"
            )
    
    return "I'm not sure how to answer that. Try asking about forecasts, growth, anomalies, or which SKU is performing best!"


# ─── Cached data loading ─────────────────────────────────────────────────────

@st.cache_data
def load_historical_data():
    data_path = "data/retail_sales.csv"
    if not os.path.exists(data_path):
        from src.generate_data import generate_retail_data
        df = generate_retail_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df

@st.cache_resource
def train_and_cache_model(df_feat, features):
    model, df_clean = train_final_lgbm(df_feat, features)
    np.random.seed(42)
    sample_indices = np.random.choice(df_clean.index, size=min(300, len(df_clean)), replace=False)
    X_sample = df_clean.loc[sample_indices, features]
    
    categorical_features = ['store_id', 'sku_id', 'temp_band']
    for col in categorical_features:
        X_sample[col] = X_sample[col].astype('category')
        
    explainer = shap.TreeExplainer(model)
    categories_dict = {
        col: df_clean[col].astype('category').cat.categories for col in categorical_features
    }
    
    return model, explainer, X_sample, categories_dict, df_clean


def main():
    # Top Navbar Bar (Stitch RetailIntel Pro aesthetic)
    st.markdown(
        """
        <div class="top-navbar">
            <div class="brand-logo">
                <div style="width: 14px; height: 14px; background-color: #5e6ad2; border-radius: 4px;"></div>
                RetailIntel Pro
            </div>
            <div class="nav-links">
                <span style="color:#f7f8f8; font-weight:600;">Analytics</span>
                <span>Inventory</span>
                <span>Supply Chain</span>
                <span>Financials</span>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div class="nav-pill-badge">Model: LightGBM v1.4 • MAPE 12.79%</div>
                <button class="export-btn">Export Data</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load raw and feature engineered data
    df_raw = load_historical_data()
    
    with st.spinner("Engineering features and compiling model..."):
        df_feat = build_feature_frame(df_raw)
        features_info = get_feature_columns()
        features = features_info['all']
        model, explainer, X_sample, categories_dict, df_clean = train_and_cache_model(df_feat, features)

    # Sidebar parameters
    st.sidebar.markdown("<div class='sidebar-section-title'>Enterprise Core</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='color:#27a644; font-size:12px; font-weight:600; margin-bottom:16px;'>🟢 AI Engine Active</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(
        """
        <div class="sidebar-nav-item">📊 Overview</div>
        <div class="sidebar-nav-item">🏬 Store Optimization</div>
        <div class="sidebar-nav-item">📦 SKU Performance</div>
        <div class="sidebar-nav-item active">⚙️ Strategic Intel</div>
        <div class="sidebar-nav-item">📑 Reports</div>
        <hr>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("### Select Product Panel")
    store_selected = st.sidebar.selectbox("Store ID", sorted(df_raw['store_id'].unique()))
    sku_selected = st.sidebar.selectbox("SKU ID", sorted(df_raw['sku_id'].unique()))
    
    # Fetch historical series for selected store/SKU
    series_df = df_feat[(df_feat['store_id'] == store_selected) & (df_feat['sku_id'] == sku_selected)].copy()
    series_df = series_df.sort_values('date').reset_index(drop=True)
    
    latest_row = series_df.iloc[-1:].copy()
    latest_date = latest_row['date'].iloc[0] if isinstance(latest_row['date'].iloc[0], str) else str(latest_row['date'].iloc[0])
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(f"### Simulation Parameters ({latest_date})")
    
    # Controls for overrides
    price_pct = st.sidebar.slider("Price Change (%)", -30.0, 30.0, 5.0, step=1.0)
    base_price = float(latest_row['price'].iloc[0])
    sim_price = base_price * (1.0 + price_pct / 100.0)
    st.sidebar.markdown(f"Simulated Price: **${sim_price:.2f}** (Base: ${base_price:.2f})")
    
    sim_promo = st.sidebar.toggle("Active Promotion", value=True)
    sim_holiday = st.sidebar.toggle("Holiday Mode", value=bool(latest_row['is_holiday'].iloc[0]))
    
    temp_override = st.sidebar.toggle("Override Temperature", value=False)
    if temp_override:
        sim_temp = st.sidebar.slider("Temperature (°C)", -5.0, 40.0, float(latest_row['temperature'].iloc[0]))
    else:
        sim_temp = float(latest_row['temperature'].iloc[0])
        
    sim_precip = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, float(latest_row['precipitation'].iloc[0]))
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.button("New Simulation", use_container_width=True)

    # Build simulation inputs
    baseline_input = latest_row[features].copy()
    scenario_input = latest_row[features].copy()
    
    scenario_input['price'] = sim_price
    scenario_input['is_promo'] = int(sim_promo)
    scenario_input['is_holiday'] = int(sim_holiday)
    scenario_input['temperature'] = sim_temp
    scenario_input['precipitation'] = sim_precip
    scenario_input['is_rainy'] = int(sim_precip > 5.0)
    
    temp_bins = [-np.inf, 5, 15, 25, np.inf]
    temp_labels = ['Cold', 'Cool', 'Mild', 'Hot']
    sim_band = pd.cut([sim_temp], bins=temp_bins, labels=temp_labels)[0]
    scenario_input['temp_band'] = sim_band

    for col in ['store_id', 'sku_id', 'temp_band']:
        baseline_input[col] = pd.Categorical([baseline_input[col].iloc[0]], categories=categories_dict[col])
        scenario_input[col] = pd.Categorical([scenario_input[col].iloc[0]], categories=categories_dict[col])

    # Predictions
    pred_base = max(0.0, float(model.predict(baseline_input)[0]))
    pred_scen = max(0.0, float(model.predict(scenario_input)[0]))
    
    pct_delta = ((pred_scen - pred_base) / pred_base) * 100 if pred_base > 0 else 0.0
    base_rev = pred_base * base_price
    scen_rev = pred_scen * sim_price
    rev_delta = scen_rev - base_rev

    # Main Grid Layout: Center Area (8 cols) + Right AI Copilot Panel (4 cols)
    main_col, copilot_col = st.columns([8, 4])
    
    with main_col:
        # KPI Cards Grid
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">BASELINE FORECAST</div>
                    <div class="metric-val">{pred_base:.1f} <span style="font-size:16px; color:#8a8f98;">units</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with kpi_col2:
            sign = "+" if pct_delta >= 0 else ""
            pill_class = "pill-green" if pct_delta >= 0 else "pill-red"
            st.markdown(
                f"""
                <div class="metric-card active-highlight">
                    <div class="metric-label" style="color:#bdc2ff;">SIMULATED FORECAST</div>
                    <div class="metric-val">{pred_scen:.1f} <span style="font-size:16px; color:#8a8f98;">units</span></div>
                    <div class="{pill_class}">📈 {sign}{pct_delta:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with kpi_col3:
            rev_sign = "+" if rev_delta >= 0 else ""
            pill_class_rev = "pill-green" if rev_delta >= 0 else "pill-red"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">REVENUE IMPACT</div>
                    <div class="metric-val">${scen_rev:,.2f}</div>
                    <div class="{pill_class_rev}">↑ {rev_sign}${rev_delta:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # AI Strategic Recommendation Card (Matching Stitch UI)
        scenario_params = {
            'price_change_pct': price_pct,
            'run_promo': sim_promo,
            'force_holiday': sim_holiday,
        }
        rec_text = generate_scenario_advice(pred_base, pred_scen, scenario_params, base_price)
        
        st.markdown(
            f"""
            <div class="ai-recommendation-card">
                <div class="ai-rec-header">
                    <span class="ai-star-icon">✦</span>
                    <span>AI STRATEGIC RECOMMENDATION</span>
                    <span style="color:#8a8f98; font-weight:400;">• Just now</span>
                </div>
                <div class="ai-rec-body">
                    {rec_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Inner Tabs Box ("WHY IS THIS SHIFTING?")
        st.markdown("<div style='color:#8a8f98; font-family:JetBrains Mono; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>WHY IS THIS SHIFTING?</div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "SHAP Attribution", "Historical Trends", 
            "Anomaly Feed", "Model Details"
        ])
        
        with tab1:
            st.markdown("#### Local Feature Attribution (SHAP Waterfall)")
            
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
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor('#0f1011')
            ax.set_facecolor('#0f1011')
            
            shap.plots.waterfall(explanation, max_display=7, show=False)
            
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('#f7f8f8')
            ax.title.set_color('#f7f8f8')
            
            st.pyplot(fig)
            plt.close()
            
            # Plain-English SHAP narration
            shap_vals = shap_values_scen[0]
            feat_names = list(features)
            abs_vals = np.abs(shap_vals)
            top_indices = np.argsort(abs_vals)[-5:][::-1]
            top_contributions = {feat_names[i]: float(shap_vals[i]) for i in top_indices}
            
            predicted_value = base_val + float(np.sum(shap_vals))
            context = {'store_id': store_selected, 'sku_id': sku_selected, 'date': latest_date}
            narration = narrate_shap_explanation(base_val, predicted_value, top_contributions, context)
            
            st.info(f"✦ **AI Explanation**: {narration}")
            
        with tab2:
            st.markdown("#### Historical Sales & Price Dynamics (Last 60 Days)")
            hist_subset = series_df.iloc[-60:].copy()
            
            fig, ax1 = plt.subplots(figsize=(10, 4))
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
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with tab3:
            st.markdown("#### Forecast Anomaly Feed")
            z_thresh = st.slider("Z-Score Threshold", 1.0, 3.0, 1.5, step=0.1, key="anomaly_z_m")
            
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
                st.info(f"No anomalies detected for {store_selected}/{sku_selected} at z ≥ {z_thresh:.1f}.")
            else:
                for _, row in anomalies.head(5).iterrows():
                    icon = "🔺" if row['z_score'] > 0 else "🔻"
                    explanation = narrate_anomaly(row)
                    st.markdown(
                        f"""
                        <div style="background-color:#0f1011; border:1px solid #23252a; border-radius:8px; padding:12px; margin-bottom:8px;">
                            <strong>{icon} {row['date']}</strong> — Forecast: {row['prediction']:.0f} vs avg {row['rolling_mean_30']:.0f} (z={row['z_score']:+.1f})<br>
                            <span style="color:#8a8f98; font-size:13px;">{explanation}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with tab4:
            st.markdown("#### Model Feature Comparison")
            comp_df = pd.DataFrame({
                'Feature': features,
                'Baseline Value': baseline_input.iloc[0].values,
                'Simulated Value': scenario_input.iloc[0].values
            })
            important_features = ['price', 'is_promo', 'is_holiday', 'temperature', 'precipitation', 'is_rainy', 'temp_band', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30']
            comp_df = comp_df[comp_df['Feature'].isin(important_features)]
            st.table(comp_df)

    # Right Column: AI Copilot Sidebar (Matching Stitch Screenshot)
    with copilot_col:
        st.markdown(
            """
            <div class="copilot-header">
                <div class="copilot-title">
                    🤖 AI Copilot
                </div>
                <span style="color:#8a8f98; font-size:12px;">Active Panel</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.caption(f"Context: Store **{store_selected}** · SKU **{sku_selected}**")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": "I've updated the dashboard with the new price change simulation. You'll notice a significant uptick in projected revenue."
                }
            ]
        
        # Render chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        user_question = st.chat_input("Ask AI about this forecast...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Simulating elasticity models..."):
                    store_ids = sorted(df_raw['store_id'].unique())
                    sku_ids = sorted(df_raw['sku_id'].unique())
                    
                    intent_data = parse_user_intent(
                        user_question, store_ids, sku_ids,
                        current_store=store_selected, current_sku=sku_selected
                    )
                    intent_data['_original_question'] = user_question
                    
                    result = execute_intent(intent_data, df_feat, features, model, all_sku_ids=sku_ids)
                    result['user_question'] = user_question
                    
                    answer = narrate_result(result, user_question=user_question)
                    
                st.markdown(answer)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
