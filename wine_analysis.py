# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Set style and parameters
plt.style.use("ggplot")
sns.set_palette("Dark2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 1. Bar Plot (Target Distribution)
plt.figure(figsize=(8, 5))
wine_counts = df['target'].value_counts()
ax = wine_counts.plot(kind='bar', color=sns.color_palette(), edgecolor='black', width=0.6)
plt.title('Distribution of Wine Classes', pad=20, fontsize=14)
plt.xlabel('Wine Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)

# Add value labels on bars
for i, count in enumerate(wine_counts):
    ax.text(i, count, str(count), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# 2. Boxplot (Target vs Proline)
sns.catplot(x="target", y="proline", data=df, kind="box", aspect=1.5)
plt.title("Proline Content by Wine Class", pad=20, fontsize=14)
plt.xlabel("Wine Class", fontsize=12)
plt.ylabel("Proline Content", fontsize=12)
plt.tight_layout()
plt.show()

# 3. Scatter Plot (Proline vs Flavanoids)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="proline", y="flavanoids", hue="target", data=df, palette="Dark2", s=80)
plt.title("Relationship Between Proline and Flavanoids by Wine Class", pad=20, fontsize=14)
plt.xlabel("Proline Content", fontsize=12)
plt.ylabel("Flavanoids Content", fontsize=12)
plt.legend(title='Wine Class')
plt.tight_layout()
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(14, 12))
corrmat = df.corr()
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Spectral_r",
                 linewidths=0.5)
plt.title("Feature Correlation Heatmap", pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Pairplot (commented out by default as it's time-consuming)
plot_cols = ['alcohol', 'malic_acid', 'ash', 'target']  
sns.pairplot(df[plot_cols], hue='target', height=2.5)
plt.suptitle('Pairwise Relationships of Selected Features', y=1.02)
plt.tight_layout()
plt.show()