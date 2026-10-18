"""
Data Visualization with Matplotlib & Seaborn
Creates histograms, box plots, bar charts, scatter plots, and a correlation heatmap.
Run: python visualization_project.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

# Built-in example dataset: Seaborn's penguins dataset is loaded online by Seaborn.
# If you have no internet, replace this with your own CSV via pd.read_csv("your_data.csv").
df = sns.load_dataset("penguins").dropna()
print("Dataset preview:")
print(df.head())
print("\\nDataset summary:")
print(df.describe(include="all"))

# 1. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="body_mass_g", bins=20, kde=True, color="#3977A8")
plt.title("Distribution of Penguin Body Mass")
plt.xlabel("Body mass (g)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "histogram.png", dpi=160)
plt.close()

# 2. Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", palette="Set2", legend=False)
plt.title("Body Mass by Penguin Species")
plt.xlabel("Species")
plt.ylabel("Body mass (g)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot.png", dpi=160)
plt.close()

# 3. Bar chart
species_counts = df["species"].value_counts().rename_axis("species").reset_index(name="count")
plt.figure(figsize=(8, 5))
sns.barplot(data=species_counts, x="species", y="count", hue="species", palette="muted", legend=False)
plt.title("Number of Penguins by Species")
plt.xlabel("Species")
plt.ylabel("Number of penguins")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bar_chart.png", dpi=160)
plt.close()

# 4. Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g",
                hue="species", style="species", s=65)
plt.title("Flipper Length vs Body Mass")
plt.xlabel("Flipper length (mm)")
plt.ylabel("Body mass (g)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_plot.png", dpi=160)
plt.close()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, square=True)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
plt.close()

print(f"\\nDone! Charts saved in: {OUTPUT_DIR.resolve()}")
