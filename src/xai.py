import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def compute_shap_values(model, X):
    """
    Computes SHAP values for a LightGBM model using TreeExplainer.
    Returns the explainer and the computed shap_values array.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # In shap >= 0.40, TreeExplainer for regression returns either a single array
    # or an array of shape (samples, features) directly. Let's inspect shape.
    if isinstance(shap_values, list):
        # Multi-class or some legacy versions return list
        shap_values = shap_values[0]
        
    return explainer, shap_values

def generate_global_importance_plot(explainer, shap_values, X, output_path="outputs/shap_summary.png"):
    """
    Generates and saves the beeswarm global feature importance plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Generate beeswarm summary plot
    # In newer shap versions, Explanation object is preferred, but summary_plot is highly backward compatible
    shap.summary_plot(shap_values, X, show=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Global feature importance plot saved to {output_path}")

def explain_single_prediction(explainer, shap_values, X, row_idx, top_n=5):
    """
    Decomposes a single prediction into base value + per-feature contributions,
    returning a formatted plain-English explanation.
    """
    row_data = X.iloc[row_idx]
    row_shap = shap_values[row_idx]
    
    # In older SHAP versions expected_value might be a list or array
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    final_pred = base_val + np.sum(row_shap)
    
    # Pair features with their SHAP values
    contributions = []
    for col, val, shap_val in zip(X.columns, row_data, row_shap):
        contributions.append({
            'feature': col,
            'value': val,
            'shap_value': shap_val,
            'abs_shap': abs(shap_val)
        })
        
    # Sort by absolute magnitude of contribution
    contributions = sorted(contributions, key=lambda x: x['abs_shap'], reverse=True)
    
    explanation_lines = [
        f"Base (average) forecast: {base_val:.2f} units",
        f"Final forecast for this row: {final_pred:.2f} units",
        "Top drivers:"
    ]
    
    for c in contributions[:top_n]:
        direction = "increased" if c['shap_value'] > 0 else "decreased"
        sign = "+" if c['shap_value'] > 0 else "-"
        explanation_lines.append(
            f"  - {c['feature']} = {c['value']} -> {direction} forecast by {abs(c['shap_value']):.2f} units ({sign}{abs(c['shap_value']):.2f})"
        )
        
    return "\n".join(explanation_lines)

def format_contributions(top_contributions):
    """
    Formats a dict of {feature_name: shap_value} into readable lines for the Gemini prompt.
    """
    lines = []
    for feat, val in top_contributions.items():
        sign = "+" if val > 0 else ""
        lines.append(f"  {feat}: {sign}{val:.1f} units")
    return "\n".join(lines)

def fallback_template_explanation(base_value, predicted_value, top_contributions):
    """
    Pure Python fallback when the Gemini API call fails.
    """
    delta = predicted_value - base_value
    pct = (delta / base_value) * 100 if base_value != 0 else 0
    direction = "above" if delta > 0 else "below"
    
    # Pick the single strongest driver
    if top_contributions:
        top_feat = max(top_contributions, key=lambda k: abs(top_contributions[k]))
        top_val = top_contributions[top_feat]
        driver_dir = "increase" if top_val > 0 else "decrease"
        return (
            f"Forecast is {abs(pct):.1f}% {direction} the baseline of {base_value:.0f} units, "
            f"mainly due to {top_feat} contributing a {abs(top_val):.0f}-unit {driver_dir}."
        )
    return f"Forecast is {abs(pct):.1f}% {direction} the baseline of {base_value:.0f} units."

def narrate_shap_explanation(base_value, predicted_value, top_contributions, context):
    """
    Uses Gemini to turn pre-computed SHAP contributions into a 1-2 sentence
    business narrative. Falls back to a template string on any API error.
    
    Parameters
    ----------
    base_value : float — average/baseline forecast from SHAP explainer
    predicted_value : float — final model prediction for this row
    top_contributions : dict — {feature_name: shap_value} for top 3-5 drivers
    context : dict — {store_id, sku_id, date} for grounding
    """
    try:
        from llm import get_model
        
        prompt = f"""You are a retail analytics assistant. Explain this sales forecast to a
business stakeholder in 1-2 plain-English sentences. Use ONLY the numbers
given below — do not estimate, round loosely, or add facts not present here.

Store: {context.get('store_id', 'N/A')}, SKU: {context.get('sku_id', 'N/A')}, Date: {context.get('date', 'N/A')}
Average/baseline forecast: {base_value:.0f} units
Actual forecast for this day: {predicted_value:.0f} units
Top contributing factors (feature -> units added or subtracted):
{format_contributions(top_contributions)}

Write the explanation now. No preamble, no bullet points, just the sentence(s)."""

        response = get_model().generate_content(prompt)
        return response.text.strip()
    except Exception:
        return fallback_template_explanation(base_value, predicted_value, top_contributions)


def generate_waterfall_plot(explainer, shap_values, X, row_idx, output_path="outputs/shap_waterfall_example.png"):
    """
    Generates and saves a waterfall plot for a single row prediction.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    row_shap = shap_values[row_idx]
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    # Build SHAP Explanation object to pass to the waterfall plotter
    explanation = shap.Explanation(
        values=row_shap,
        base_values=base_val,
        data=X.iloc[row_idx].values,
        feature_names=list(X.columns)
    )
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Waterfall plot for row {row_idx} saved to {output_path}")

if __name__ == "__main__":
    from features import build_feature_frame, get_feature_columns
    from models import train_final_lgbm
    
    input_path = "data/retail_sales.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run generate_data.py first.")
    else:
        # Load and build features
        print("Loading data...")
        df = pd.read_csv(input_path)
        df_feat = build_feature_frame(df)
        
        features = get_feature_columns()['all']
        
        # Train model
        print("Training final model...")
        model, df_clean = train_final_lgbm(df_feat, features)
        
        # We'll use a subset of data for SHAP computation to keep it fast
        # Select 500 random rows from df_clean for explanation
        print("Computing SHAP values on subset of data...")
        np.random.seed(42)
        sample_indices = np.random.choice(df_clean.index, size=500, replace=False)
        X_sample = df_clean.loc[sample_indices, features]
        
        # Convert categories to category dtype for explainer (if not already done)
        categorical_features = ['store_id', 'sku_id', 'temp_band']
        for col in categorical_features:
            X_sample[col] = X_sample[col].astype('category')
            
        explainer, shap_values = compute_shap_values(model, X_sample)
        
        # Generate Global Summary
        print("Generating global importance plot...")
        generate_global_importance_plot(explainer, shap_values, X_sample)
        
        # Find a row where is_promo == 1 to demonstrate local explanation
        promo_rows = X_sample[X_sample['is_promo'] == 1]
        if not promo_rows.empty:
            row_idx = X_sample.index.get_loc(promo_rows.index[0])
        else:
            row_idx = 0
            
        # Generate Waterfall
        print(f"Generating waterfall plot for row {row_idx}...")
        generate_waterfall_plot(explainer, shap_values, X_sample, row_idx)
        
        # Print local explanation text
        print("\nLocal Explanation Summary:")
        explanation = explain_single_prediction(explainer, shap_values, X_sample, row_idx)
        print(explanation)
