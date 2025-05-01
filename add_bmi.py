import pandas as pd

df = pd.read_csv("CategoricalDataset.csv")


df['BMI'] = (df['Weight'] / (df['Height'] ** 2)).round(2)


df.to_csv("CategoricalDataset.csv", index=False)

