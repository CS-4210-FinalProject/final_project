import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

#For top 5
#chosen_features = ['Weight', 'Height', 'family_history_with_overweight', 'Gender', 'FAVC']
#For top 10
#chosen_features = ['Weight', 'Height', 'family_history_with_overweight', 'Gender', 'FAVC', 'CAEC', 'SCC', 'CH20', 'Age', 'CALC']
#For top 15
chosen_features = ['Weight', 'Height', 'family_history_with_overweight', 'Gender', 'FAVC', 'CAEC', 'SCC', 'CH20', 'Age', 'CALC', 'NCP', 'MTRANS', 'FAF', 'TUE', 'SMOKE']

#Encode categorical variables for all
'''
categorical = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
'''

categorical = [col for col in chosen_features if col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']]
numerical = [col for col in chosen_features if col in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']]

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

# Initialize and train perceptron
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train, y_train)

# Evaluate
y_pred = perceptron.predict(X_test)
print("Perceptron Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    early_stopping=True)

mlp.fit(X_train, y_train)

# Evaluate
y_pred_mlp = mlp.predict(X_test)
print("\nMLPClassifier Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp, target_names=le.classes_))

#need predicted probabilities for brier score
y_train_proba = mlp.predict_proba(X_train)
y_val_proba = mlp.predict_proba(X_val)
y_test_proba = mlp.predict_proba(X_test)

#Function to calculate multiclass Brier score
def multiclass_brier_score(y_true, y_proba):
    n_classes = y_proba.shape[1]
    true_labels = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_proba - true_labels) ** 2, axis=1))

#Calculate Brier scores
train_brier = multiclass_brier_score(y_train, y_train_proba)
val_brier = multiclass_brier_score(y_val, y_val_proba)
test_brier = multiclass_brier_score(y_test, y_test_proba)

print("\nBrier Scores:")
print(f"Training set: {train_brier:.4f}")
print(f"Validation set: {val_brier:.4f}")
print(f"Test set: {test_brier:.4f}")

# Create DataFrame with predictions and probabilities
results_df = pd.DataFrame({
    'True_Label': le.inverse_transform(y_test),
    'Perceptron_Prediction': le.inverse_transform(y_pred),
    'MLP_Prediction': le.inverse_transform(y_pred_mlp)
})
proba_df = pd.DataFrame(y_test_proba, columns=[f'Prob_{cls}' for cls in le.classes_])
results_df = pd.concat([results_df, proba_df], axis=1)

# Save to CSV
results_df.to_csv('test_set_predictions.csv', index=False)
print("\nSaved test set predictions to 'test_set_predictions.csv'")

# Confusion Matrices
labels = sorted(results_df['True_Label'].unique())

# Perceptron Confusion Matrix
cm_perceptron = confusion_matrix(results_df['True_Label'], results_df['Perceptron_Prediction'], labels=labels)
disp_perceptron = ConfusionMatrixDisplay(confusion_matrix=cm_perceptron, display_labels=labels)
disp_perceptron.plot(xticks_rotation=45, cmap='Purples')
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title("Perceptron Confusion Matrix")
plt.tight_layout()
plt.show()

# MLP Confusion Matrix
cm_mlp = confusion_matrix(results_df['True_Label'], results_df['MLP_Prediction'], labels=labels)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=labels)
disp_mlp.plot(xticks_rotation=45, cmap='Blues')
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title("MLP Confusion Matrix")
plt.tight_layout()
plt.show()

# PCA on class probabilities
proba_cols = [col for col in results_df.columns if col.startswith('Prob_')]
pca = PCA(n_components=2)
pca_result = pca.fit_transform(results_df[proba_cols])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=results_df['True_Label'],
    style=results_df['MLP_Prediction'],
    palette='Set2',
    s=60
)
plt.title("PCA of MLP Prediction Probabilities")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA on the features from the test set (X_test) instead of the entire feature set (X)
pca_features = PCA(n_components=2)
pca_features_result = pca_features.fit_transform(X_test)  # Use X_test to match with test set predictions

# Make sure to use the correct data for the hue and style parameters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_features_result[:, 0],
    y=pca_features_result[:, 1],
    hue=le.inverse_transform(y_test),  # Use true labels from y_test
    style=le.inverse_transform(y_pred_mlp),  # Use MLP predictions
    palette='Set1',
    s=60
)
plt.title("PCA of Features (Test Set)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# t-SNE on class probabilities
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(results_df[proba_cols])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    hue=results_df['True_Label'],
    style=results_df['MLP_Prediction'],
    palette='Dark2',
    s=60
)
plt.title("t-SNE of MLP Prediction Probabilities")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()