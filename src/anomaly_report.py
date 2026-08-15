import os
import numpy as np
import pandas as pd

def detect_anomalies(feat_df, predictions_df, z_threshold=1.5):
    """
    Flags days where the forecast deviates from that series' own rolling
    mean by more than z_threshold standard deviations. Pure pandas/numpy —
    no LLM involved in detection, only in the write-up that follows.
    """
    merged = predictions_df.merge(
        feat_df[['date', 'store_id', 'sku_id', 'rolling_mean_30', 'rolling_std_30',
                 'is_holiday', 'is_promo', 'temperature', 'precipitation', 'is_weekend']],
        on=['date', 'store_id', 'sku_id'],
        how='left'
    )
    
    # Guard against zero std
    merged['z_score'] = (
        (merged['prediction'] - merged['rolling_mean_30'])
        / merged['rolling_std_30'].replace(0, np.nan)
    )
    
    anomalies = merged[merged['z_score'].abs() >= z_threshold].copy()
    anomalies = anomalies.sort_values('z_score', ascending=False).reset_index(drop=True)
    return anomalies

def narrate_anomaly(row):
    """
    Uses Gemini to explain a single detected anomaly row using ONLY the
    context flags present in the data. Falls back to a template string
    on any API error.
    """
    try:
        from llm import get_model
        
        prompt = f"""You are a retail demand analyst writing one line for a daily report. Explain
this forecasted anomaly using ONLY the fields given — do not invent a cause
that isn't represented in the data below.

Store: {row['store_id']}, SKU: {row['sku_id']}, Date: {row['date']}
Forecast: {row['prediction']:.0f} units vs 30-day average {row['rolling_mean_30']:.0f} units ({row['z_score']:+.1f} std deviations)
Context flags: is_holiday={row['is_holiday']}, is_promo={row['is_promo']}, temperature={row['temperature']}°C, precipitation={row['precipitation']}mm, is_weekend={row['is_weekend']}

Write one sentence explaining the likely driver(s), grounded only in the flags above.
If none of the flags obviously explain it, say the anomaly's cause is unclear from available signals rather than guessing."""

        response = get_model().generate_content(prompt)
        return response.text.strip()
    except Exception:
        # Fallback: build explanation from flags
        drivers = []
        if row.get('is_holiday') == 1:
            drivers.append("holiday")
        if row.get('is_promo') == 1:
            drivers.append("active promotion")
        if row.get('is_weekend') == 1:
            drivers.append("weekend")
        if row.get('precipitation', 0) > 10:
            drivers.append(f"heavy rain ({row['precipitation']:.0f}mm)")
            
        direction = "above" if row['z_score'] > 0 else "below"
        if drivers:
            cause = ", ".join(drivers)
            return (
                f"{row['date']}: forecast is {row['z_score']:+.1f} std devs {direction} "
                f"the 30-day average — likely driven by {cause}."
            )
        return (
            f"{row['date']}: forecast is {row['z_score']:+.1f} std devs {direction} "
            f"the 30-day average — cause unclear from available signals."
        )

def generate_daily_report(feat_df, predictions_df, z_threshold=1.5, max_rows=10):
    """
    Generates a markdown report of the top anomalies with AI-narrated explanations.
    """
    anomalies = detect_anomalies(feat_df, predictions_df, z_threshold=z_threshold)
    
    if anomalies.empty:
        return "No forecast anomalies detected above the threshold."
    
    # Limit to top N anomalies by absolute z-score
    anomalies['abs_z'] = anomalies['z_score'].abs()
    anomalies = anomalies.nlargest(max_rows, 'abs_z')
    
    report_lines = [f"### Forecast Anomaly Report ({len(anomalies)} flagged)\n"]
    
    for _, row in anomalies.iterrows():
        explanation = narrate_anomaly(row)
        direction_icon = "🔺" if row['z_score'] > 0 else "🔻"
        report_lines.append(
            f"{direction_icon} **{row['store_id']} / {row['sku_id']}** — "
            f"{row['prediction']:.0f} units (z={row['z_score']:+.1f})  \n"
            f"_{explanation}_\n"
        )
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    from features import build_feature_frame, get_feature_columns
    from models import train_final_lgbm
    
    input_path = "data/retail_sales.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run generate_data.py first.")
    else:
        print("Loading data and building features...")
        df = pd.read_csv(input_path)
        df_feat = build_feature_frame(df)
        features = get_feature_columns()['all']
        
        print("Training model...")
        model, df_clean = train_final_lgbm(df_feat, features)
        
        # Generate predictions on the clean dataset
        import lightgbm as lgb
        preds = model.predict(df_clean[features])
        preds = np.clip(preds, 0, None)
        
        predictions_df = df_clean[['date', 'store_id', 'sku_id']].copy()
        predictions_df['prediction'] = preds
        
        print("Detecting anomalies...")
        anomalies = detect_anomalies(df_feat, predictions_df, z_threshold=1.5)
        print(f"Found {len(anomalies)} anomalies")
        
        # Show the top 5
        if not anomalies.empty:
            top5 = anomalies.nlargest(5, 'z_score')
            for _, row in top5.iterrows():
                explanation = narrate_anomaly(row)
                print(f"\n{row['store_id']}/{row['sku_id']} on {row['date']}: "
                      f"{row['prediction']:.0f} vs avg {row['rolling_mean_30']:.0f} (z={row['z_score']:+.1f})")
                print(f"  → {explanation}")
