# Importing libraries for data analysis and visualization
import pandas as pd                               # For working with datasets (loading, cleaning, etc.)
import numpy as np                                # For numerical computations
import matplotlib.pyplot as plt                   # For creating basic plots
import seaborn as sns                             # For creating advanced and attractive plots

# Importing libraries for machine learning tasks
#from sklearn.impute import SimpleImputer         # For handling missing values #SimpleImputer — Creates a tool for replacing missing values.
from sklearn.preprocessing import StandardScaler  # For scaling features
from sklearn.decomposition import PCA             # For dimensionality reduction

#1. load the Dataset
df = pd.read_csv("data/data.csv")
print(df.head()) #head displays the first five rows by default
#print(df.columns)

#2. Exploratory Data Analysis (EDA)
#2.1 Structured EDA Workflow
#Step 1: Frame the question

#question: How do the numerical characteristics of cell nuclei differ between benign and malignant tumors, and can PCA summarize the variation in these measurements using fewer dimensions?
#1. EDA: What are the main differences in cell nucleus measurements between benign and malignant tumors?

#2. PCA: Can we reduce the 30 numerical features to fewer principal components while keeping most of the information about how the measurements vary?

target_counts = df['diagnosis'].value_counts()
target_percentages = (target_counts / len(df) * 100)
print("\nCounts:")
print(target_counts)
print("\nPercentages:")
print(target_percentages)

#the dataset is reasonably balanced, with higher proportion of benign
#step2: Data Overview and Feature Selection
# Check the shape of the dataset
print("Shape of the dataset (rows, columns):", df.shape)
# Display column names and their data types
print("\nColumn names and data types:")
print(df.dtypes.to_string())

# Identify non-feature columns to drop
non_feature_columns = {"id", "diagnosis", "Unnamed: 32"}  #we exclude diagnosis Because it is the label that tells us whether a tumor is benign or malignant. We want to investigate how the other 30 features relate to this label
features = df.columns.difference(non_feature_columns)

# Identify numerical features
numeric_features = df[features].select_dtypes(
    include="number"
).columns.tolist()

# Step 3: Missing Values and Simple Fixes

# Count missing values for each column
missing = df.isnull().sum()

# Keep only columns that actually have missing values
missing = missing[missing > 0]

# Create a summary DataFrame with count and percentage of missing values
missing_info = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': (missing / len(df)) * 100
})


# Display missing value summary
print("Missing Values Summary:")
print(missing_info)

# Step 4: Univariate Checks for Selected Features

# Choose 4 features to analyze
selected_features = ["radius_mean", "texture_mean", "area_mean", "perimeter_mean"]

# Loop through each selected feature and calculate basic statistics (round to 2 decimal places)
for col in selected_features:
    print(f"\n--- {col} ---")
    print(f"Mean   : {df[col].mean():.2f}")
    print(f"Median : {df[col].median():.2f}")
    print(f"Min    : {df[col].min():.2f}")
    print(f"Max    : {df[col].max():.2f}")

# Set up histogram plotting
plt.figure(figsize=(15, 4))

# Plot histograms for each feature
for i, col in enumerate(selected_features, 1):
    plt.subplot(1, 4, i)   #plt.subplot(rows, columns, position)
    df[col].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 5: Bivariate Analysis

# Features to check
median_by_diagnosis = df.groupby("diagnosis")[selected_features].median()
print("\nMedian values by diagnosis:")
print(median_by_diagnosis)

plt.figure(figsize=(12, 8))

for i, col in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)

    sns.boxplot(
        data=df,
        x="diagnosis",
        y=col,
        order=["B", "M"]
    )

    plt.title(f"{col} by Diagnosis")
    plt.xlabel("Diagnosis (B / M)")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Step 6: Multivariate Analysis
corr_matrix = df[numeric_features].corr()
print("\nCorrelation Matrix:")
#print(corr_matrix.round(2))

# Set up the figure
plt.figure(figsize=(12, 10))

# Plot the heatmap
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Correlation Matrix of Numerical Features")

plt.tight_layout()
plt.show()


# Find and print highly correlated pairs (|correlation| ≥ 0.7)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) >= 0.7:
            high_corr_pairs.append((col1, col2, corr_val))

# Print results
print("\nHighly correlated feature pairs (|correlation| ≥ 0.7):")
for col1, col2, val in high_corr_pairs:
    print(f"{col1} ↔ {col2} : correlation = {val:.2f}")

# Step 7: Principal Component Analysis (PCA)

# Select the 30 numerical features
X = df[numeric_features]

# Create scaler
scaler = StandardScaler()

# Standardize the features
X_scaled = scaler.fit_transform(X)

print("Original shape:", X.shape)
print("Scaled shape:", X_scaled.shape)
#Notice that standardization does not reduce the number of features. It only changes their scale.

#create PCA object
pca = PCA()

#Fit PCA on standardized data
X_PCA = pca.fit_transform(X_scaled)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_

print("Explained variance for each component:")
print(explained_variance)

# Cumulative explained variance
cumulative_var = explained_variance.cumsum()
print("\nCumulative explained variance:", cumulative_var)

# Plot the cumulative explained variance (scree plot)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), cumulative_var, marker="o", color="blue")
plt.axhline(y=0.8, color="r", linestyle="--", label="80% Threshold")
plt.axhline(y=0.9, color="g", linestyle="--", label="90% Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot (Elbow Curve)")
plt.legend()
plt.show()

# Scatter plot of PC1 vs PC2:

# Convert diagnosis labels to numbers for coloring
y = df["diagnosis"].map({"B": 0, "M": 1})

# Scatter plot of PC1 vs PC2
plt.figure(figsize=(8, 6))

plt.scatter(
    X_PCA[:, 0],
    X_PCA[:, 1],
    c=y,
    cmap="coolwarm",
    alpha=0.6,
    s=20
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: PC1 vs PC2 (colored by Diagnosis)")

cbar = plt.colorbar(ticks=[0, 1])
cbar.set_ticklabels(["Benign", "Malignant"])
cbar.set_label("Diagnosis")

plt.show()

# Step 8: Communicate Results
# Examine PCA component weights
loadings = pd.DataFrame(
    pca.components_.T,
    index=numeric_features,
    columns=[f"PC{i+1}" for i in range(len(numeric_features))]
)

# Show strongest absolute weights for PC1
print("\nTop contributing features for PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).round(2).head(5))

print(
    loadings["PC1"]
    .reindex(loadings["PC1"].abs().sort_values(ascending=False).index)
    .head(5)

)