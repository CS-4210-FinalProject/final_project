import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CategoricalDataset.csv")

# Drop classification column 
df = df.drop("NObeyesdad", axis=1)

# PCA cannot run with null values, drop them
df1 = df.dropna()

scaling = StandardScaler() 

# Use fit and transform method 
scaling.fit(df1)
Scaled_data = scaling.transform(df1)

# Set the n_component=1
principal = PCA(n_components=1)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)

# Check the values of the eigen vectors produced by principal components
print("Eigen vectors produced by principal components:\n",principal.components_)

# Let's assume `df1.columns` holds the names of your original features.
# Replace this with your actual column names.
feature_names = df1.columns.tolist()

# For each principal component, sort the features by the absolute value of their loadings
for i, component in enumerate(np.array(principal.components_)):
    sorted_idx = np.argsort(np.abs(component))[::-1]  # Sort in descending order of absolute value
    sorted_features = [feature_names[idx] for idx in sorted_idx]
    print(f"\nTop 5 Most meaningful features for Principal Component {i + 1}: {sorted_features[:5]}")  # Top 5 features
    print(f"Top 10 Most meaningful features for Principal Component {i + 1}: {sorted_features[:10]}")  # Top 5 features
    print(f"Top 15 Most meaningful features for Principal Component {i + 1}: {sorted_features[:15]}")  # Top 5 features
    print(f"Ranking of meainingful features {i + 1}: {sorted_features[:10]}")  # Top 5 features

# Each principal component represents a different direction of data variation, with the first component capturing the largest portion of the variance. Subsequent components capture progressively less variance, but still contribute to the overall explanation of the data. 


# Features with higher absolute values contribute more to the principal component.