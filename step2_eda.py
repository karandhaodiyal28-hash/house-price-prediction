"""
Step 2: Exploratory Data Analysis (EDA)
Generates comprehensive visualisations and summary statistics.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Font setup for cross-platform compatibility
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC', 'Sarasa Mono SC']
plt.rcParams['axes.unicode_minus'] = False

# ── Load cleaned data ─────────────────────────────────────────────
df = pd.read_csv("cleaned_house_prices.csv")
print("=" * 60)
print("STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 2a. Summary statistics ──────────────────────────────────────
print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

# Separate location dummies for count
location_cols = [c for c in df.columns if c.startswith("Location_")]
other_cols = [c for c in df.columns if not c.startswith("Location_")]

# ── 2b. Distribution of Price ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["Price"], bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Distribution of House Prices", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Frequency")

sns.boxplot(y=df["Price"], ax=axes[1], color="#4C72B0")
axes[1].set_title("Box Plot of House Prices", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Price ($)")

plt.tight_layout()
plt.savefig("eda_01_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: eda_01_price_distribution.png")

# ── 2c. Correlation heatmap ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[other_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 10})
ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_02_correlation_heatmap.png")

# ── 2d. Scatter plots — Price vs numerical features ─────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = ["Area", "Bedrooms", "Bathrooms", "Parking"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for ax, feat, clr in zip(axes.flatten(), features, colors):
    ax.scatter(df[feat], df["Price"], alpha=0.4, s=20, color=clr, edgecolors="none")
    # Trend line
    z = np.polyfit(df[feat], df["Price"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_line, p(x_line), color="black", linewidth=2, linestyle="--")
    ax.set_title(f"Price vs {feat}", fontsize=13, fontweight="bold")
    ax.set_xlabel(feat)
    ax.set_ylabel("Price ($)")

plt.suptitle("Price vs Numerical Features", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("eda_03_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_03_scatter_plots.png")

# ── 2e. Location-wise price analysis ────────────────────────────
if location_cols:
    loc_counts = df[location_cols].sum().sort_values(ascending=False)
    loc_means = {}
    for col in location_cols:
        loc_name = col.replace("Location_", "")
        loc_means[loc_name] = df.loc[df[col] == 1, "Price"].mean()
    loc_series = pd.Series(loc_means).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(loc_counts.index.str.replace("Location_", ""), loc_counts.values, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Number of Houses by Location", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Location")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(loc_series.index, loc_series.values, color="#DD8452", edgecolor="white")
    axes[1].set_title("Average Price by Location", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Location")
    axes[1].set_ylabel("Average Price ($)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("eda_04_location_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: eda_04_location_analysis.png")

# ── 2f. Pairplot ────────────────────────────────────────────────
selected = ["Area", "Bedrooms", "Bathrooms", "Parking", "Price"]
pp = sns.pairplot(df[selected], diag_kind="kde", plot_kws={"alpha": 0.3, "s": 15},
                  corner=True)
pp.figure.suptitle("Pairplot of Numerical Features", fontsize=14, fontweight="bold", y=1.02)
plt.savefig("eda_05_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_05_pairplot.png")

# ── 2g. Bedrooms & Bathrooms count plots ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="Bedrooms", data=df, ax=axes[0], palette="Blues_d")
axes[0].set_title("Bedroom Count Distribution", fontsize=13, fontweight="bold")

sns.countplot(x="Bathrooms", data=df, ax=axes[1], palette="Oranges_d")
axes[1].set_title("Bathroom Count Distribution", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("eda_06_bed_bath_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_06_bed_bath_counts.png")

print("\nEDA complete. All plots saved.")