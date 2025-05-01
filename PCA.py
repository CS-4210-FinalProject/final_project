import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CategoricalDataset.csv")

# Drop classification column 
df = df.drop("NObeyesdad", axis=1)
print(df.columns)

# PCA cannot run with null values, drop them
df1 = df.dropna()

scaling = StandardScaler() 

# Use fit and transform method 
scaling.fit(df1)
Scaled_data = scaling.transform(df1)

# Set the n_component=3
principal = PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)

# Check the shape of the data after PCA 
print(x.shape)

# Check the values of the eigen vectors produced by principal components
print(principal.components_)


plt.figure(figsize=(10,10))
plt.scatter(x[:,0], x[:,1], c = df['NOObeyesdad'], cmap = 'plasma')

# Scale
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df1)

# # Run PCA
# pca = PCA(n_components=2) # 3 components would also deliver the same results 
# df_pca = pca.fit_transform(df_scaled)

# print(df_pca)
# print(df.shape)
# print(pca.components_)



#y = df.loc[df.index, 'NObeyesdad']

# plt.figure(figsize=(8,6))
# scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c = y, cmap = 'viridis', alpha = 0.7)
# plt.show()
