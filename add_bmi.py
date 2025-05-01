import pandas as pd

df = pd.read_csv("ObesityDataset.csv")
df1 = pd.read_csv("CategoricalDataset.csv")
df1 = df1.drop("BMI", axis = 1)

bmiSeries = (df['Weight'] / (df['Height'] ** 2)).round(2)
print(bmiSeries)
df1.insert(3, "BMI", bmiSeries)

# Categorize the BMI values
# 1 = underweight 
# 2 = normal weight 
# 3 = overweight 
# 4 = extremely overweight
df1['BMI'] = pd.cut(x=df1['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[1, 2, 3, 4])

df1.to_csv("CategoricalDataset.csv", index=False)

