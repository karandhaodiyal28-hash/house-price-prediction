# House Price Prediction — Data Science Mini Project

## Overview
A complete end-to-end data science project that predicts house prices using Linear Regression.

## Dataset Features
| Feature   | Description                        |
|-----------|------------------------------------|
| Area      | Total area in square feet          |
| Bedrooms  | Number of bedrooms                 |
| Bathrooms | Number of bathrooms                |
| Parking   | Number of parking spots            |
| Location  | Area type (Downtown/Suburb/Rural/Semi-Urban/Coastal) |
| Price     | Target variable — house price ($)  |

## Project Structure

```
house_price_prediction/
├── README.md                        # This file
├── run_project.py                   # Master script (runs all steps)
├── generate_dataset.py              # Step 0: Generate synthetic dataset
├── step1_data_cleaning.py           # Step 1: Clean data
├── step2_eda.py                     # Step 2: Exploratory Data Analysis
├── step3_model_training.py          # Step 3: Train, predict, evaluate
├── step4_business_insights.py       # Step 4: Generate 10 business insights
│
├── house_prices.csv                 # Raw dataset (with noise)
├── cleaned_house_prices.csv         # Cleaned dataset
├── predictions.csv                  # Model predictions on test set
├── house_price_model.pkl            # Trained Linear Regression model
├── business_insights.txt            # 10 actionable business insights
│
├── eda_01_price_distribution.png    # Price histogram + boxplot
├── eda_02_correlation_heatmap.png   # Correlation matrix
├── eda_03_scatter_plots.png         # Price vs numerical features
├── eda_04_location_analysis.png     # Location-wise price analysis
├── eda_05_pairplot.png              # Pairplot of all numerical features
├── eda_06_bed_bath_counts.png       # Bedroom & bathroom distributions
├── model_evaluation.png             # Actual vs Predicted + Residuals
└── feature_coefficients.png         # Linear Regression coefficients
```

## Steps Performed

### Step 0 — Dataset Generation
Creates a synthetic dataset of 500 houses with realistic relationships, injected missing values, duplicates, and outliers.

### Step 1 — Data Cleaning
- Removed 5 duplicate rows
- Filled missing values (median for numerical, mode for categorical)
- Removed outliers using IQR method (1 Area outlier, 8 Price outliers)
- One-hot encoded the Location feature
- Final clean dataset: **491 rows × 10 columns**

### Step 2 — Exploratory Data Analysis (EDA)
- Descriptive statistics
- Price distribution (histogram + boxplot)
- Correlation heatmap
- Scatter plots (Price vs Area, Bedrooms, Bathrooms, Parking)
- Location-wise count and average price analysis
- Pairplot of all numerical features
- Bedroom and bathroom count distributions

### Step 3 — Model Training & Evaluation
- **Train/Test Split**: 80/20 (392 train, 99 test)
- **Model**: Linear Regression
- **Results on Test Set**:
  - MAE  = $116,874.58
  - RMSE = $166,533.44
  - R²   = 0.7184

### Step 4 — Business Insights
10 actionable insights covering price ranges, location premiums, feature importance, economies of scale, optimal configurations, and investment strategies.

## How to Run

### Option 1: Run everything at once
```bash
python run_project.py
```

### Option 2: Run step by step
```bash
python generate_dataset.py
python step1_data_cleaning.py
python step2_eda.py
python step3_model_training.py
python step4_business_insights.py
```

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```