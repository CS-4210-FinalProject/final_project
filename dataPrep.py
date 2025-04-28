import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Set Pandas option to not hide any columns
pd.set_option('display.max_columns', None)
df = pd.read_csv('ObesityDataset.csv', sep=',', header=0) #reading the data by using Pandas library

#Checking to see how many unique values there are
'''
for col in df.columns:
    unique_vals = df[col].unique()
    print(f"Unique values in '{col}': {unique_vals}\n")
'''

#Separate into features and truth value
rawX = df.drop('NObeyesdad', axis=1)
rawY = df['NObeyesdad']

#Encode categorical variables
categorical = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

#Label the obesity levels
le = LabelEncoder()
Y = le.fit_transform(rawY)

#Process categorical features categoricalX
catX = rawX[categorical].apply(LabelEncoder().fit_transform)

#Process numerical features so that it centers the data and normalizes spread
scaler = StandardScaler()
numX = pd.DataFrame(scaler.fit_transform(rawX[numerical]), columns=numerical)

#Recombine the new features
X = pd.concat([numX, catX], axis=1)

#Split data for test set which will be 20%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y  # maintains class distribution
)

#Separate training and validation from remaining 60% and 20% respectively
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,  # 0.25 x 0.8 = 0.2
    random_state=42,
    stratify=y_train_val
)

#Checking sizes to make sure everything is awesome
print(f"Train size: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
print(f"Validation size: {len(X_val)} samples ({len(X_val)/len(X):.1%})")
print(f"Test size: {len(X_test)} samples ({len(X_test)/len(X):.1%})")