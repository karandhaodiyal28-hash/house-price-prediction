"""
Step 3: Model Training, Prediction & Evaluation
Splits data, trains a Linear Regression model, predicts prices,
and evaluates using MAE, RMSE, and R² Score.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Font setup
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC', 'Sarasa Mono SC']
plt.rcParams['axes.unicode_minus'] = False

# ── Load cleaned data ─────────────────────────────────────────────
df = pd.read_csv("cleaned_house_prices.csv")
print("=" * 60)
print("STEP 3 — MODEL TRAINING, PREDICTION & EVALUATION")
print("=" * 60)

# ── 3a. Separate features (X) and target (y) ────────────────────
X = df.drop("Price", axis=1)
y = df["Price"]
print(f"\nFeature columns : {list(X.columns)}")
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# ── 3b. Train-Test Split (80/20) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set  : {X_train.shape[0]} rows")
print(f"Testing set   : {X_test.shape[0]} rows")

# ── 3c. Train Linear Regression ─────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# ── 3d. Predictions ─────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ── 3e. Evaluation Metrics ─────────────────────────────────────
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"  MAE  = ${mae:,.2f}")
    print(f"  RMSE = ${rmse:,.2f}")
    print(f"  R²   = {r2:.4f}")
    return mae, rmse, r2

print("\n" + "=" * 40)
print("EVALUATION METRICS")
print("=" * 40)
train_mae, train_rmse, train_r2 = evaluate(y_train, y_pred_train, "Training Set")
test_mae, test_rmse, test_r2 = evaluate(y_test, y_pred_test, "Testing Set")

# ── 3f. Feature Coefficients ────────────────────────────────────
print("\n--- Feature Coefficients ---")
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)
print(coef_df.to_string(index=False))

# ── 3g. Visualisations ──────────────────────────────────────────
# Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=25, color="#4C72B0", edgecolors="none")
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_title("Actual vs Predicted Prices (Test Set)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].legend()

# Residuals
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=25, color="#C44E52", edgecolors="none")
axes[1].axhline(y=0, color="black", linewidth=2, linestyle="--")
axes[1].set_title("Residual Plot (Test Set)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: model_evaluation.png")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4C72B0" if c >= 0 else "#C44E52" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
ax.set_title("Feature Coefficients (Linear Regression)", fontsize=14, fontweight="bold")
ax.set_xlabel("Coefficient Value")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_coefficients.png")

# ── 3h. Save sample predictions to CSV ─────────────────────────
results_df = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": y_pred_test.round(0),
    "Difference": (y_test.values - y_pred_test).round(0),
    "Abs_Error": np.abs(y_test.values - y_pred_test).round(0),
})
results_df.to_csv("predictions.csv", index=False)
print("Saved: predictions.csv")

# ── 3i. Save model ──────────────────────────────────────────────
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: house_price_model.pkl")

print("\nStep 3 complete!")