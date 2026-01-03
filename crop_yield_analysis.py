#!/usr/bin/env python3
"""
Crop Yield Prediction Analysis
Machine learning project for predicting crop yields using rainfall, temperature, and pesticide data
"""

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load yield data
df_yield = pd.read_csv("data/yield.csv")

print("Yield data shape:", df_yield.shape)
print("\nFirst 10 rows:")
print(df_yield.head(10))

# Rename columns
df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
print("\nAfter renaming:")
print(df_yield.head())

# Drop unnecessary columns
df_yield = df_yield.drop(['Year Code', 'Element Code', 'Element', 'Area Code', 
                          'Domain Code', 'Domain', 'Unit', 'Item Code'], axis=1)
print("\nAfter dropping columns:")
print(df_yield.head())
print("\nDataset description:")
print(df_yield.describe())
print("\nDataset info:")
print(df_yield.info())

# Load rainfall data
df_rain = pd.read_csv("data/rainfall.csv")
print("\nRainfall data:")
print(df_rain.head())
print(df_rain.tail())

# Rename columns to remove leading space
df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})
print("\nRainfall info:")
print(df_rain.info())

# Convert rainfall to numeric
df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(
    df_rain['average_rain_fall_mm_per_year'], errors='coerce')
print("\nAfter converting to numeric:")
print(df_rain.info())

# Drop missing values
df_rain = df_rain.dropna()
print("\nRainfall description:")
print(df_rain.describe())

# Merge yield and rainfall data
yield_df = pd.merge(df_yield, df_rain, on=['Year', 'Area'])
print("\nAfter merging with rainfall:")
print(yield_df.head())
print(yield_df.describe())

# Load pesticides data
df_pes = pd.read_csv("data/pesticides.csv")
print("\nPesticides data:")
print(df_pes.head())

# Rename and drop columns
df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
df_pes = df_pes.drop(['Element', 'Domain', 'Unit', 'Item'], axis=1)
print("\nPesticides after processing:")
print(df_pes.head())
print(df_pes.describe())
print(df_pes.info())

# Merge with pesticides data
yield_df = pd.merge(yield_df, df_pes, on=['Year', 'Area'])
print("\nDataset shape after pesticides merge:", yield_df.shape)
print(yield_df.head())

# Load temperature data
avg_temp = pd.read_csv("data/temp.csv")
print("\nTemperature data:")
print(avg_temp.head())
print(avg_temp.describe())

# Rename columns
avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country": 'Area'})
print("\nAfter renaming temperature columns:")
print(avg_temp.head())

# Merge with temperature data
yield_df = pd.merge(yield_df, avg_temp, on=['Area', 'Year'])
print("\nFinal dataset:")
print(yield_df.head())
print("Shape:", yield_df.shape)
print("\nDescription:")
print(yield_df.describe())

# Check for missing values
print("\nMissing values:")
print(yield_df.isnull().sum())

# Group by Item
print("\nCount by Item:")
print(yield_df.groupby('Item').count())

print("\nUnique areas:", yield_df['Area'].nunique())

# Top 10 areas by yield
print("\nTop 10 areas by total yield:")
print(yield_df.groupby(['Area'], sort=True)['hg/ha_yield'].sum().nlargest(10))

# Top 10 item-area combinations
print("\nTop 10 item-area combinations:")
print(yield_df.groupby(['Item', 'Area'], sort=True)['hg/ha_yield'].sum().nlargest(10))

# Correlation analysis
correlation_data = yield_df.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(correlation_data, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")

# One-hot encoding
print("\nApplying one-hot encoding...")
yield_df_onehot = pd.get_dummies(yield_df, columns=['Area', "Item"], prefix=['Country', "Item"])

features = yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
label = yield_df['hg/ha_yield']

print("\nFeatures before dropping Year:")
print(features.head())

# Drop Year column
features = features.drop(['Year'], axis=1)
print("\nFeatures info:")
print(features.info())
print(features.head())

# Scale features
print("\nScaling features...")
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
print("Scaled features shape:", features.shape)

# Split data
print("\nSplitting data into train and test sets...")
train_data, test_data, train_labels, test_labels = train_test_split(
    features, label, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Drop Year from one-hot encoded dataframe for later use
yield_df_onehot = yield_df_onehot.drop(['Year'], axis=1)
print("\nOne-hot encoded dataframe:")
print(yield_df_onehot.head())

# Prepare test dataframe
test_df = pd.DataFrame(
    test_data, 
    columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns
)

# Extract country and item information
cntry = test_df[[col for col in test_df.columns if 'Country' in col]].stack()[
    test_df[[col for col in test_df.columns if 'Country' in col]].stack() > 0]
cntrylist = list(pd.DataFrame(cntry).index.get_level_values(1))
countries = [i.split("_")[1] for i in cntrylist]

itm = test_df[[col for col in test_df.columns if 'Item' in col]].stack()[
    test_df[[col for col in test_df.columns if 'Item' in col]].stack() > 0]
itmlist = list(pd.DataFrame(itm).index.get_level_values(1))
items = [i.split("_")[1] for i in itmlist]

print("\nTest dataframe before dropping encoded columns:")
print(test_df.head())

# Drop encoded columns
test_df.drop([col for col in test_df.columns if 'Item' in col], axis=1, inplace=True)
test_df.drop([col for col in test_df.columns if 'Country' in col], axis=1, inplace=True)
print("\nTest dataframe after cleanup:")
print(test_df.head())

# Add country and item back
test_df['Country'] = countries
test_df['Item'] = items
print("\nTest dataframe with Country and Item:")
print(test_df.head())

# Train Decision Tree model
print("\nTraining Decision Tree Regressor...")
clf = DecisionTreeRegressor()
model = clf.fit(train_data, train_labels)

# Make predictions
test_df["yield_predicted"] = model.predict(test_data)
test_df["yield_actual"] = pd.DataFrame(test_labels)["hg/ha_yield"].tolist()

test_group = test_df.groupby("Item")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(test_df["yield_actual"], test_df["yield_predicted"], edgecolors=(0, 0, 0), alpha=0.6)
ax.set_xlabel('Actual Yield (hg/ha)', fontsize=12)
ax.set_ylabel('Predicted Yield (hg/ha)', fontsize=12)
ax.set_title("Actual vs Predicted Crop Yield", fontsize=14)
ax.plot([test_df["yield_actual"].min(), test_df["yield_actual"].max()],
        [test_df["yield_actual"].min(), test_df["yield_actual"].max()],
        'r--', lw=2, label='Perfect Prediction')
ax.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

# Feature importance
varimp = {
    'imp': model.feature_importances_,
    'names': yield_df_onehot.columns[yield_df_onehot.columns != "hg/ha_yield"]
}

# Plot all feature importances
a4_dims = (8.27, 16.7)
fig, ax = plt.subplots(figsize=a4_dims)
df_imp = pd.DataFrame.from_dict(varimp)
df_imp.sort_values(ascending=False, by=["imp"], inplace=True)
df_imp = df_imp.dropna()
sns.barplot(x="imp", y="names", palette="vlag", data=df_imp, orient="h", ax=ax)
ax.set_title('Feature Importance (All Features)', fontsize=14)
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_all.png')
print("All feature importance plot saved as 'feature_importance_all.png'")

# Plot top 7 feature importances
a4_dims = (16.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
df_imp = pd.DataFrame.from_dict(varimp)
df_imp.sort_values(ascending=False, by=["imp"], inplace=True)
df_imp = df_imp.dropna()
df_imp = df_imp.nlargest(7, 'imp')
sns.barplot(x="imp", y="names", palette="vlag", data=df_imp, orient="h", ax=ax)
ax.set_title('Top 7 Most Important Features', fontsize=14)
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_top7.png')
print("Top 7 feature importance plot saved as 'feature_importance_top7.png'")

# Box plot by Item
a4_dims = (16.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x="Item", y="hg/ha_yield", palette="vlag", data=yield_df, ax=ax)
ax.set_title('Crop Yield Distribution by Item', fontsize=14)
ax.set_xlabel('Crop Type', fontsize=12)
ax.set_ylabel('Yield (hg/ha)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('yield_by_item_boxplot.png')
print("Yield by item boxplot saved as 'yield_by_item_boxplot.png'")

print("\n=== Analysis Complete ===")
print(f"Model trained on {len(train_data)} samples")
print(f"Model tested on {len(test_data)} samples")
print(f"Number of features: {features.shape[1]}")
print(f"Output files generated:")
print("  - correlation_heatmap.png")
print("  - actual_vs_predicted.png")
print("  - feature_importance_all.png")
print("  - feature_importance_top7.png")
print("  - yield_by_item_boxplot.png")
