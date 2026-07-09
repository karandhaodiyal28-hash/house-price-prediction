"""
Step 0: Generate a synthetic House Price Dataset.
Run this first to create 'house_prices.csv'.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

locations = ["Downtown", "Suburb", "Rural", "Semi-Urban", "Coastal"]
loc_price_multiplier = {
    "Downtown": 1.8,
    "Suburb": 1.0,
    "Rural": 0.6,
    "Semi-Urban": 1.2,
    "Coastal": 1.5,
}

area = np.random.randint(500, 5000, size=n)
bedrooms = np.random.randint(1, 6, size=n)
bathrooms = np.random.randint(1, 4, size=n)
parking = np.random.randint(0, 4, size=n)
location = np.random.choice(locations, size=n)

# Base price formula with realistic relationships
base_price = (
    area * 150
    + bedrooms * 50000
    + bathrooms * 30000
    + parking * 20000
)
price = base_price * np.array([loc_price_multiplier[loc] for loc in location])
noise = np.random.normal(0, price * 0.10, size=n)  # 10% noise
price = (price + noise).astype(int)

df = pd.DataFrame(
    {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Location": location,
        "Price": price,
    }
)

# Inject some realistic dirty data (missing values, outliers, duplicates)
# ~5% missing values spread across columns
for col in ["Area", "Bedrooms", "Bathrooms", "Parking", "Price"]:
    mask = np.random.random(n) < 0.05
    df.loc[mask, col] = np.nan

# A few duplicate rows
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)

# A couple of outlier rows (extreme area)
df.loc[0, "Area"] = 99999
df.loc[1, "Price"] = 9999999

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("house_prices.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())