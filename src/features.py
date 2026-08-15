import pandas as pd
import numpy as np

def add_calendar_features(df):
    """
    Adds calendar and cyclical calendar features.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Cyclical encodings
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def add_exogenous_features(df):
    """
    Adds exogenous features like promo, precipitation category, temperature band,
    and proximity to the next holiday.
    """
    df = df.copy()
    
    # 1. Promo flag is directly available as promo_flag
    df['is_promo'] = df['promo_flag'].astype(int)
    
    # 2. Rainy flag (precipitation threshold > 5.0mm)
    df['is_rainy'] = (df['precipitation'] > 5.0).astype(int)
    
    # 3. Temperature band (categorical bucket)
    # Define bins: Cold (<5), Cool (5-15), Mild (15-25), Hot (>25)
    df['temp_band'] = pd.cut(
        df['temperature'],
        bins=[-np.inf, 5, 15, 25, np.inf],
        labels=['Cold', 'Cool', 'Mild', 'Hot']
    )
    
    # 4. Proximity to next holiday (days to next holiday, capped at 30 days)
    # Extract unique holiday dates (where is_holiday == 1)
    df['date'] = pd.to_datetime(df['date'])
    holiday_dates = df[df['is_holiday'] == 1]['date'].unique()
    holiday_dates = sorted(pd.to_datetime(holiday_dates))
    
    def calculate_days_to_next_holiday(date):
        future_holidays = [h for h in holiday_dates if h >= date]
        if not future_holidays:
            return 30 # Cap at 30 if no future holidays in dataset
        days = (future_holidays[0] - date).days
        return min(days, 30)
    
    # Apply calculation (vectorized or mapping over unique dates for performance)
    unique_dates = df['date'].unique()
    date_to_days = {d: calculate_days_to_next_holiday(pd.Timestamp(d)) for d in unique_dates}
    df['days_to_next_holiday'] = df['date'].map(date_to_days).astype(float)
    
    return df

def add_lag_features(df, target='sales'):
    """
    Adds lag features (lag_1, lag_7, lag_14, lag_30) grouped by store and SKU.
    """
    df = df.copy()
    lags = [1, 7, 14, 30]
    
    # Group by store_id and sku_id to prevent leakages across time series
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_id', 'sku_id'])[target].shift(lag)
        
    return df

def add_rolling_features(df, target='sales'):
    """
    Adds rolling mean and rolling std features grouped by store and SKU.
    CRITICAL: The target is shifted by 1 first to prevent leakage of the current day's target.
    """
    df = df.copy()
    windows = [7, 30]
    
    # Group by store_id and sku_id, shift target by 1 first
    df['temp_shifted_target'] = df.groupby(['store_id', 'sku_id'])[target].shift(1)
    
    # Group again on the shifted target to compute rolling features within groups
    for window in windows:
        df[f'rolling_mean_{window}'] = (
            df.groupby(['store_id', 'sku_id'])['temp_shifted_target']
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        df[f'rolling_std_{window}'] = (
            df.groupby(['store_id', 'sku_id'])['temp_shifted_target']
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )
        
    df = df.drop(columns=['temp_shifted_target'])
    return df

def get_feature_columns():
    """
    Single source of truth for the complete feature set used by the models.
    """
    categorical = ['store_id', 'sku_id', 'temp_band']
    numeric = [
        'price', 'is_promo', 'temperature', 'precipitation', 'is_holiday',
        'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year',
        'is_weekend', 'is_month_start', 'is_month_end',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_rainy', 'days_to_next_holiday'
    ]
    lags = ['lag_1', 'lag_7', 'lag_14', 'lag_30']
    rolling = ['rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']
    
    return {
        'categorical': categorical,
        'numeric': numeric,
        'lags': lags,
        'rolling': rolling,
        'all': categorical + numeric + lags + rolling
    }

def build_feature_frame(df, target='sales'):
    """
    Executes the entire feature engineering pipeline in sequence.
    """
    df = df.copy()
    df = add_calendar_features(df)
    df = add_exogenous_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    return df

if __name__ == "__main__":
    import os
    input_path = "data/retail_sales.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run generate_data.py first.")
    else:
        print("Loading data...")
        df = pd.read_csv(input_path)
        print("Building feature frame...")
        df_feat = build_feature_frame(df)
        
        # Verify columns
        feat_info = get_feature_columns()
        print(f"Total features created: {len(feat_info['all'])}")
        
        # Verify NaNs only appear at the beginning of each group
        max_lag_or_window = 30
        print("\nChecking for missing values in feature frame:")
        nan_means = df_feat[feat_info['all']].isna().mean()
        print(nan_means[nan_means > 0])
        
        # Group checks
        print("\nVerifying NaNs are restricted to early records per group:")
        for name, group in df_feat.groupby(['store_id', 'sku_id']):
            # Within each group of length ~1095 days, the first 30 rows can have NaNs for lags/rolling
            early_nans = group.iloc[:max_lag_or_window][feat_info['lags'] + feat_info['rolling']].isna().sum().sum()
            late_nans = group.iloc[max_lag_or_window:][feat_info['lags'] + feat_info['rolling']].isna().sum().sum()
            print(f"  Group {name}: Early NaNs = {early_nans}, Late NaNs = {late_nans}")
            assert late_nans == 0, f"Error: Found late NaNs in group {name}!"
            
        print("\nFeature verification passed successfully!")
