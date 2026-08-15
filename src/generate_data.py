import os
import numpy as np
import pandas as pd

def generate_retail_data(
    start_date="2023-01-01",
    end_date="2025-12-31",
    stores=["STORE_01", "STORE_02", "STORE_03"],
    skus=["SKU_01", "SKU_02", "SKU_03", "SKU_04", "SKU_05"],
    seed=42
):
    """
    Generates a synthetic panel dataset of retail sales with realistic trends,
    seasonality, price elasticity, promotions, weather, and holidays.
    """
    np.random.seed(seed)
    
    # Try to import holidays, fallback to hardcoded if not installed
    try:
        import holidays
        us_holidays = holidays.US(years=range(2023, 2026))
    except ImportError:
        us_holidays = {
            pd.Timestamp("2023-01-01"): "New Year's Day",
            pd.Timestamp("2023-11-23"): "Thanksgiving",
            pd.Timestamp("2023-12-25"): "Christmas Day",
            pd.Timestamp("2024-01-01"): "New Year's Day",
            pd.Timestamp("2024-11-28"): "Thanksgiving",
            pd.Timestamp("2024-12-25"): "Christmas Day",
            pd.Timestamp("2025-01-01"): "New Year's Day",
            pd.Timestamp("2025-11-27"): "Thanksgiving",
            pd.Timestamp("2025-12-25"): "Christmas Day",
        }

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)
    
    # Base characteristics for each store
    store_base = {
        "STORE_01": {"demand_multiplier": 1.2, "weather_sensitivity": 0.8},
        "STORE_02": {"demand_multiplier": 0.9, "weather_sensitivity": 1.2},
        "STORE_03": {"demand_multiplier": 1.0, "weather_sensitivity": 1.0},
    }
    
    # Base characteristics for each SKU
    sku_base = {
        "SKU_01": {"base_price": 4.99, "elasticity": -1.5, "base_demand": 100, "summer_peak": True},
        "SKU_02": {"base_price": 14.99, "elasticity": -0.8, "base_demand": 40, "summer_peak": False},
        "SKU_03": {"base_price": 1.99, "elasticity": -2.0, "base_demand": 250, "summer_peak": True},
        "SKU_04": {"base_price": 29.99, "elasticity": -1.2, "base_demand": 20, "summer_peak": False},
        "SKU_05": {"base_price": 9.99, "elasticity": -1.0, "base_demand": 70, "summer_peak": False},
    }

    records = []
    
    # Weather simulation (common across region but with some random fluctuation)
    # Annual temperature cycle (peaking in July, coldest in January)
    day_of_year = date_range.dayofyear
    temp_trend = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 105) / 365) # Peak ~July (day 196)
    temp_noise = np.random.normal(0, 3, n_days)
    temperatures = temp_trend + temp_noise
    
    # Precipitation: rainy days (15% probability)
    precipitation = np.where(np.random.rand(n_days) < 0.15, np.random.exponential(10, n_days), 0.0)

    for store in stores:
        s_mult = store_base[store]["demand_multiplier"]
        s_sens = store_base[store]["weather_sensitivity"]
        
        for sku in skus:
            sku_info = sku_base[sku]
            base_d = sku_info["base_demand"]
            base_p = sku_info["base_price"]
            elasticity = sku_info["elasticity"]
            
            # Price simulation: baseline price with occasional promotional markdowns
            # Let's say promo happens randomly ~10% of the time per store-SKU
            promo_flag = np.random.rand(n_days) < 0.10
            prices = np.where(promo_flag, base_p * 0.8, base_p) # 20% discount on promo
            
            for i, date in enumerate(date_range):
                # 1. Base trend: slow growth (2% per year)
                years_since_start = (date - pd.Timestamp(start_date)).days / 365.25
                trend = 1.0 + 0.02 * years_since_start
                
                # 2. Price effect (elasticity)
                price_ratio = prices[i] / base_p
                price_effect = price_ratio ** elasticity
                
                # 3. Weekly seasonality
                # 0=Monday, 6=Sunday
                dow = date.dayofweek
                if dow in [5, 6]:  # Weekend
                    weekly_seasonality = 1.3  # +30% sales
                elif dow == 4:     # Friday
                    weekly_seasonality = 1.15 # +15% sales
                else:
                    weekly_seasonality = 0.9  # weekday dip
                
                # 4. Yearly seasonality
                doy = date.dayofyear
                if sku_info["summer_peak"]:
                    yearly_seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * (doy - 100) / 365)
                else:
                    yearly_seasonality = 1.0 + 0.15 * np.cos(2 * np.pi * (doy - 20) / 365)
                
                # 5. Promo boost (in addition to price effect)
                promo_boost = 1.4 if promo_flag[i] else 1.0
                
                # 6. Weather effects
                temp_dev = temperatures[i] - 15.0 # deviation from 15C
                if sku_info["summer_peak"]:
                    temp_effect = 1.0 + 0.01 * temp_dev * s_sens
                else:
                    temp_effect = 1.0 - 0.005 * temp_dev * s_sens
                
                # Rain suppression (if precipitation > 5mm, footfall dips)
                rain_effect = 1.0
                if precipitation[i] > 5.0:
                    rain_effect = max(0.6, 1.0 - 0.02 * precipitation[i] * s_sens)
                
                # 7. Holiday effect
                is_hol = 0
                hol_name = "None"
                date_obj = date.to_pydatetime().date() if hasattr(date, "to_pydatetime") else date
                if date in us_holidays or date_obj in us_holidays:
                    is_hol = 1
                    hol_name = us_holidays[date] if date in us_holidays else us_holidays[date_obj]
                
                holiday_boost = 1.6 if is_hol == 1 else 1.0
                
                # Compute expected sales (multiplicative process)
                expected_sales = (
                    base_d * s_mult * trend * price_effect * weekly_seasonality *
                    yearly_seasonality * promo_boost * temp_effect * rain_effect * holiday_boost
                )
                
                # Heteroskedastic noise (noise variance scales with expected sales)
                noise = np.random.normal(0, 0.12 * expected_sales)
                sales = max(0.0, expected_sales + noise)
                
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store,
                    "sku_id": sku,
                    "sales": round(sales, 2),
                    "price": round(prices[i], 2),
                    "promo_flag": int(promo_flag[i]),
                    "temperature": round(temperatures[i], 1),
                    "precipitation": round(precipitation[i], 2),
                    "is_holiday": is_hol,
                    "holiday_name": str(hol_name)
                })
                
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Generating synthetic retail demand dataset...")
    df = generate_retail_data()
    os.makedirs("data", exist_ok=True)
    output_path = "data/retail_sales.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully written to {output_path}")
    print(f"Total records: {len(df)}")
    print(df.head())
