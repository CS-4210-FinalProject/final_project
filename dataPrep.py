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

df.to_csv("CategoricalDataset.csv")

