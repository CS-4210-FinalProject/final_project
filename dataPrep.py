import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import the necessary dataset 
df = pd.read_csv("ObesityDataset.csv")

# BMI Formula (vikafitnessguide.com)
df['BMI'] = df['Weight']/(df['Height']**2)
print(df[['Weight', 'BMI']].round(2))

# Now select only numeric columns for scaling
df = df.select_dtypes(include=[np.number])

# Scale the data before applying PCA 
scaling = StandardScaler()

# Use fit and transform method 
scaling.fit(df)
scaled_data = scaling.transform(df)

# Set the n_components to 3 
principal=PCA(n_components=3)
principal.fit(scaled_data)
x=principal.transform(scaled_data)

# Check the dimensions of the data after PCA
print(x.shape)
