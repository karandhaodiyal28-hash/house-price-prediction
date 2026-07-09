"""
Master Script — House Price Prediction Mini Project
Runs all steps in sequence: Generate → Clean → EDA → Model → Insights
"""
import subprocess
import sys

steps = [
    ("Step 0: Generate Dataset", "generate_dataset.py"),
    ("Step 1: Data Cleaning", "step1_data_cleaning.py"),
    ("Step 2: Exploratory Data Analysis", "step2_eda.py"),
    ("Step 3: Model Training & Evaluation", "step3_model_training.py"),
    ("Step 4: Business Insights", "step4_business_insights.py"),
]

print("=" * 60)
print("  HOUSE PRICE PREDICTION — MINI PROJECT")
print("=" * 60)

for name, script in steps:
    print(f"\n>>> Running {name} ...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {name}:")
        print(result.stderr)
        sys.exit(1)

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nOutput files:")
print("  - house_prices.csv          (raw dataset)")
print("  - cleaned_house_prices.csv  (cleaned dataset)")
print("  - predictions.csv           (model predictions)")
print("  - house_price_model.pkl     (trained model)")
print("  - business_insights.txt     (10 business insights)")
print("  - eda_01_price_distribution.png")
print("  - eda_02_correlation_heatmap.png")
print("  - eda_03_scatter_plots.png")
print("  - eda_04_location_analysis.png")
print("  - eda_05_pairplot.png")
print("  - eda_06_bed_bath_counts.png")
print("  - model_evaluation.png")
print("  - feature_coefficients.png")