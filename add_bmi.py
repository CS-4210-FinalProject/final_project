import pandas as pd

df = pd.read_csv("ObesityDataset.csv")

bmiSeries = (df['Weight'] / (df['Height'] ** 2)).round(2)
df.insert(3, "BMI", bmiSeries)

df['BMI'] = pd.cut(x=df['BMI'], bins=[0.11, 0.8325, 1.555, 2.277, 3], labels=[1, 2, 3, 4])

df.to_csv("ObesityDataset1.csv", index=False)

