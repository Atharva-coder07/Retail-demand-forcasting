import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Add the project root to the path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import build_feature_frame, get_feature_columns
from src.models import train_final_lgbm
from src.xai import compute_shap_values

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
    </style>
    """,
    unsafe_allow_html=True
)

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
    
    return model, explainer, X_sample, categories_dict

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
        model, explainer, X_sample, categories_dict = train_and_cache_model(df_feat, features)

    # Sidebar parameters
    st.sidebar.markdown("### Select Product Panel")
    store_selected = st.sidebar.selectbox("Store ID", sorted(df_raw['store_id'].unique()))
    sku_selected = st.sidebar.selectbox("SKU ID", sorted(df_raw['sku_id'].unique()))
    
    # Fetch historical series for selected store/SKU
    series_df = df_feat[(df_feat['store_id'] == store_selected) & (df_feat['sku_id'] == sku_selected)].copy()
    series_df = series_df.sort_values('date').reset_index(drop=True)
    
    # Get the latest row for "What-If" simulation
    latest_row = series_df.iloc[-1:].copy()
    latest_date = latest_row['date'].iloc[0].strftime('%Y-%m-%d')
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(f"### What-If Parameters (Simulating {latest_date})")
    
    # Controls for overrides
    price_pct = st.sidebar.slider("Price Change (%)", -30.0, 30.0, 0.0, step=1.0)
    sim_price = float(latest_row['price'].iloc[0] * (1.0 + price_pct / 100.0))
    st.sidebar.markdown(f"Simulated Price: **${sim_price:.2f}** (Base: ${latest_row['price'].iloc[0]:.2f})")
    
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
        price_diff = sim_price - latest_row['price'].iloc[0]
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

    # Tabs for visualization
    tab1, tab2, tab3 = st.tabs(["Explainable AI (SHAP)", "Historical Trends", "Model Details"])
    
    with tab1:
        st.markdown("### Why did the forecast change? (Local SHAP Attribution)")
        st.write("The waterfall plot below decomposes the simulated prediction from the average baseline model forecast.")
        
        # Calculate local SHAP values for the scenario
        # TreeExplainer expects features matching training configuration
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
        # Customize plot appearance for dark mode compatibility
        fig.patch.set_facecolor('#0f1011')
        ax.set_facecolor('#0f1011')
        
        shap.plots.waterfall(explanation, max_display=7, show=False)
        
        # Tweak colors for dark mode readability
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#f7f8f8')
        ax.title.set_color('#f7f8f8')
        
        st.pyplot(fig)
        plt.close()
        
    with tab2:
        st.markdown("### Historical Sales & Pricing Trend")
        
        # Plot last 60 days of actuals for context
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
        
        # Second axis for price
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
        
        # Show recent values
        st.markdown("#### Latest Features State (Baseline vs Simulated)")
        comp_df = pd.DataFrame({
            'Feature': features,
            'Baseline Value': baseline_input.iloc[0].values,
            'Simulated Value': scenario_input.iloc[0].values
        })
        # Filter only changed rows or important ones for readability
        important_features = ['price', 'is_promo', 'is_holiday', 'temperature', 'precipitation', 'is_rainy', 'temp_band', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30']
        comp_df = comp_df[comp_df['Feature'].isin(important_features)]
        st.table(comp_df)

if __name__ == "__main__":
    main()
