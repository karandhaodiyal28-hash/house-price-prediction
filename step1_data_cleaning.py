"""
Step 1: Data Cleaning
Loads raw data, handles missing values, outliers, duplicates,
and encodes categorical features. Saves cleaned dataset.
"""
import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv("house_prices.csv")
print("=" * 60)
print("STEP 1 — DATA CLEANING")
print("=" * 60)

print(f"\nOriginal shape : {df.shape}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nDuplicate rows : {df.duplicated().sum()}")

# ── 1. Drop duplicates ───────────────────────────────────────────
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nAfter dropping duplicates: {df.shape}")

# ── 2. Handle missing values ─────────────────────────────────────
# Numerical columns — fill with median
num_cols = ["Area", "Bedrooms", "Bathrooms", "Parking", "Price"]
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} missing values with median = {median_val}")

# Categorical columns — fill with mode
cat_cols = ["Location"]
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled {col} missing values with mode = '{mode_val}'")

# ── 3. Remove outliers using IQR on 'Area' and 'Price' ──────────
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = data.shape[0]
    data = data[(data[column] >= lower) & (data[column] <= upper)]
    after = data.shape[0]
    print(f"  {column}: removed {before - after} outliers  (bounds: {lower:.0f}–{upper:.0f})")
    return data

print("\nOutlier removal (IQR method):")
df = remove_outliers_iqr(df, "Area")
df = remove_outliers_iqr(df, "Price")

# ── 4. Encode categorical feature (Location) — One-Hot ──────────
df = pd.get_dummies(df, columns=["Location"], drop_first=False, dtype=int)

# ── 5. Final check ───────────────────────────────────────────────
print(f"\nFinal shape    : {df.shape}")
print(f"Remaining nulls: {df.isnull().sum().sum()}")
print(f"Columns        : {list(df.columns)}")

# ── Save cleaned data ────────────────────────────────────────────
df.to_csv("cleaned_house_prices.csv", index=False)
print("\nCleaned dataset saved to 'cleaned_house_prices.csv'")
print("\nSample rows:")
print(df.head())