import pandas as pd 

df = pd.read_csv("ObesityDataset.csv")
print(df.columns)

df = df.dropna()
# Ages range from 14-61 

for i in range(len(df['Gender'])):
    
    if df.loc[i, 'Gender'] == "Male": 
        df.loc[i, 'Gender'] = 1
    else: 
        df.loc[i, 'Gender'] = 2
    print(df.loc[i, 'Gender'])

# Categories - "Adolescent", "Young Adult", "Middle Aged", "Geriatric"
df['Age'] = pd.cut(x=df['Age'], bins=[14, 20, 35, 47, 63], labels=[1, 2, 3, 4])


df['Weight'] = pd.cut(x=df['Weight'], bins=[39, 67, 95, 123, 151], labels=[1, 2, 3, 4])
df['Height'] = pd.cut(x=df['Height'], bins=[1.45, 1.58, 1.71, 1.98], labels=[1, 2, 3])

for i in range(len(df['family_history_with_overweight'])):
    
    if df.loc[i, 'family_history_with_overweight'] == "yes": 
        df.loc[i, 'family_history_with_overweight'] = 1
    else: 
        df.loc[i, 'family_history_with_overweight'] = 0
    print(df.loc[i, 'family_history_with_overweight'])


for i in range(len(df['SCC'])):
    
    if df.loc[i, 'SCC'] == "yes": 
        df.loc[i, 'SCC'] = 1
    else: 
        df.loc[i, 'SCC'] = 0
    print(df.loc[i, 'SCC'])

for i in range(len(df['FAVC'])):
    
    if df.loc[i, 'FAVC'] == "yes": 
        df.loc[i, 'FAVC'] = 1
    else: 
        df.loc[i, 'FAVC'] = 0
    print(df.loc[i, 'FAVC'])


for i in range(len(df['CAEC'])):
    
    if df.loc[i, 'CAEC'] == "no": 
        df.loc[i, 'CAEC'] = 0
    elif df.loc[i, 'CAEC'] == "Sometimes": 
        df.loc[i, 'CAEC'] = 1
    elif df.loc[i, 'CAEC'] == "Frequently":
        df.loc[i, 'CAEC'] = 2
    elif df.loc[i, 'CAEC'] == "Always":
        df.loc[i, 'CAEC'] = 3
        
    print(df.loc[i, 'CAEC'])

for i in range(len(df['CALC'])):
    
    if df.loc[i, 'CALC'] == "no": 
        df.loc[i, 'CALC'] = 0
    elif df.loc[i, 'CALC'] == "Sometimes": 
        df.loc[i, 'CALC'] = 1
    elif df.loc[i, 'CALC'] == "Frequently":
        df.loc[i, 'CALC'] = 2
    elif df.loc[i, 'CALC'] == "Always":
        df.loc[i, 'CALC'] = 3
        
    print(df.loc[i, 'CAEC'])

for i in range(len(df['SMOKE'])):
    
    if df.loc[i, 'SMOKE'] == "yes": 
        df.loc[i, 'SMOKE'] = 1
    else: 
        df.loc[i, 'SMOKE'] = 0
    print(df.loc[i, 'SMOKE'])


for i in range(len(df['MTRANS'])):
    
    if df.loc[i, 'MTRANS'] == "Public_Transportation": 
        df.loc[i, 'MTRANS'] = 0
    elif df.loc[i, 'MTRANS'] == "Walking": 
        df.loc[i, 'MTRANS'] = 1
    elif df.loc[i, 'MTRANS'] == "Automobile":
        df.loc[i, 'MTRANS'] = 2
    elif df.loc[i, 'MTRANS'] == "Motorbike":
        df.loc[i, 'MTRANS'] = 3
    elif df.loc[i, 'MTRANS'] == "Bike":
        df.loc[i, 'MTRANS'] = 4
    print(df.loc[i, 'MTRANS'])

for i in range(len(df['NObeyesdad'])):
    
    if df.loc[i, 'NObeyesdad'] == "Insufficient_Weight": 
        df.loc[i, 'NObeyesdad'] = 1
    elif df.loc[i, 'NObeyesdad'] == "Normal_Weight": 
        df.loc[i, 'NObeyesdad'] = 2
    elif df.loc[i, 'NObeyesdad'] == "Overweight_Level_I":
        df.loc[i, 'NObeyesdad'] = 3
    elif df.loc[i, 'NObeyesdad'] == "Overweight_Level_II":
        df.loc[i, 'NObeyesdad'] = 4
    elif df.loc[i, 'NObeyesdad'] == "Obesity_Type_I":
        df.loc[i, 'NObeyesdad'] = 5
    elif df.loc[i, 'NObeyesdad'] == "Obesity_Type_II":
        df.loc[i, 'NObeyesdad'] = 6
    elif df.loc[i, 'NObeyesdad'] == "Obesity_Type_III":
        df.loc[i, 'NObeyesdad'] = 7
    print(df.loc[i, 'NObeyesdad'])


# BMI Formula (vikafitnessguide.com)
df['BMI'] = df['Weight']/(df['Height']**2)
print(df[['Weight', 'BMI']].round(2))


df = df.dropna()
df.to_csv("CategoricalDataset.csv", index = False)


# create an attribute for BMI (weight/height)
print(df['Height'].max())
print(df['Height'].min()) 