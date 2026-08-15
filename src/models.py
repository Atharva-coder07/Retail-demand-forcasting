import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Try to import prophet, fallback gracefully if not installed
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

def mape(y_true, y_pred, epsilon=1e-5):
    """
    Computes Mean Absolute Percentage Error (MAPE).
    Guards against division-by-zero by ignoring/masking actual sales that are 0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mask where y_true is zero (or very close to zero)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return 0.0
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def naive_baseline_predict(df, target='sales'):
    """
    Baseline: Predict sales_t = sales_{t-1} per store-SKU group.
    """
    return df.groupby(['store_id', 'sku_id'])[target].shift(1)

def seasonal_naive_baseline_predict(df, target='sales'):
    """
    Baseline: Predict sales_t = sales_{t-7} per store-SKU group.
    """
    return df.groupby(['store_id', 'sku_id'])[target].shift(7)

def time_series_cv_scores(df, features, target='sales', n_splits=5):
    """
    Performs leakage-safe TimeSeriesSplit Cross Validation on LightGBM.
    Ensures store_id, sku_id, and temp_band are treated as categorical features.
    Clips predictions at 0 (sales cannot be negative).
    """
    df = df.copy()
    
    # Ensure categorical features are of category type for LightGBM
    categorical_features = ['store_id', 'sku_id', 'temp_band']
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Sort chronologically by date to preserve temporal order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Drop rows that contain NaNs in features (due to lag/rolling window startup periods)
    df_clean = df.dropna(subset=features + [target]).reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_mapes = []
    
    print(f"\nRunning {n_splits}-fold Time-Series Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_clean)):
        train_data = df_clean.iloc[train_idx]
        val_data = df_clean.iloc[val_idx]
        
        # Prepare training and validation sets
        X_train, y_train = train_data[features], train_data[target]
        X_val, y_val = val_data[features], val_data[target]
        
        # Define LightGBM Dataset
        train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_features)
        
        # LightGBM Parameters
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'n_estimators': 150,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        # Predict on validation set
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None) # Clip negative predictions to 0
        
        # Compute MAPE
        fold_mape = mape(y_val, preds)
        fold_mapes.append(fold_mape)
        print(f"  Fold {fold + 1} MAPE: {fold_mape:.2f}%")
        
    avg_mape = np.mean(fold_mapes)
    print(f"Average LightGBM CV MAPE: {avg_mape:.2f}%")
    return fold_mapes, avg_mape

def train_final_lgbm(df, features, target='sales'):
    """
    Trains a final production LightGBM model on all available history.
    """
    df = df.copy()
    
    categorical_features = ['store_id', 'sku_id', 'temp_band']
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    df = df.sort_values('date').reset_index(drop=True)
    df_clean = df.dropna(subset=features + [target]).reset_index(drop=True)
    
    X = df_clean[features]
    y = df_clean[target]
    
    train_dataset = lgb.Dataset(X, label=y, categorical_feature=categorical_features)
    
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    
    # Train final model on all data
    model = lgb.train(params, train_dataset)
    return model, df_clean

def train_prophet_for_series(series_df, horizon=30, price_col='price'):
    """
    Fits Facebook Prophet for an individual store-SKU series.
    Returns the Prophet model and forecast dataframe.
    """
    if not HAS_PROPHET:
        raise ImportError(
            "Prophet is not installed. Please install it using `pip install prophet` "
            "to use the Prophet forecasting features."
        )
        
    # Prepare Prophet format (ds, y)
    prophet_df = series_df[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Optional regressor (price)
    if price_col in series_df.columns:
        prophet_df['price'] = series_df[price_col].values
        
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    if price_col in series_df.columns:
        model.add_regressor('price')
        
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=horizon)
    
    # Since future dataframe won't have price, we need to populate it.
    # Simple strategy: use the last known price for future predictions.
    if price_col in series_df.columns:
        last_price = series_df[price_col].iloc[-1]
        future_prices = list(series_df[price_col].values) + [last_price] * horizon
        future['price'] = future_prices
        
    forecast = model.predict(future)
    forecast['yhat'] = np.clip(forecast['yhat'], 0, None) # Clip negative forecasts to 0
    
    return model, forecast

if __name__ == "__main__":
    import os
    from features import build_feature_frame, get_feature_columns
    
    input_path = "data/retail_sales.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run generate_data.py first.")
    else:
        # Load and build features
        print("Loading data...")
        df = pd.read_csv(input_path)
        df_feat = build_feature_frame(df)
        
        # Calculate Baselines
        print("\nCalculating Baseline Models...")
        df_feat['pred_naive'] = naive_baseline_predict(df_feat)
        df_feat['pred_seasonal_naive'] = seasonal_naive_baseline_predict(df_feat)
        
        # Filter rows without NaNs in baselines for a fair comparison
        df_eval_base = df_feat.dropna(subset=['pred_naive', 'pred_seasonal_naive', 'sales'])
        
        naive_mape = mape(df_eval_base['sales'], df_eval_base['pred_naive'])
        s_naive_mape = mape(df_eval_base['sales'], df_eval_base['pred_seasonal_naive'])
        
        print(f"Naive Baseline MAPE: {naive_mape:.2f}%")
        print(f"Seasonal Naive Baseline (t-7) MAPE: {s_naive_mape:.2f}%")
        
        # Run Cross-Validation for LightGBM
        features = get_feature_columns()['all']
        fold_scores, avg_lgb_mape = time_series_cv_scores(df_feat, features)
        
        # Verification check
        print("\nEvaluating model performance vs baselines:")
        if avg_lgb_mape < s_naive_mape:
            print("  SUCCESS: LightGBM outperforms the Seasonal Naive baseline.")
        else:
            print("  WARNING: LightGBM does not beat the Seasonal Naive baseline. Check features for leakage/bugs.")
            
        if s_naive_mape < naive_mape:
            print("  SUCCESS: Seasonal Naive baseline outperforms the Naive baseline.")
        else:
            print("  WARNING: Seasonal Naive is worse than Naive baseline. Double check date alignment.")
            
        print("\nTraining final LightGBM production model...")
        model, df_clean = train_final_lgbm(df_feat, features)
        print("Model trained successfully.")
