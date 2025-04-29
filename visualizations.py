import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# === Load & prepare data ===
df = pd.read_csv('ObesityDataset.csv')
le = LabelEncoder()
df['class_code'] = le.fit_transform(df['NObeyesdad'])
class_order = list(le.classes_)

# === Make output directory ===
outdir = "Visualizations_Output"
os.makedirs(outdir, exist_ok=True)

# === 1) Distribution of classes ===
plt.figure(figsize=(8,4))
counts = df['NObeyesdad'].value_counts().reindex(class_order)
plt.bar(class_order, counts.values)
plt.xticks(rotation=45)
plt.xlabel('Obesity Class')
plt.ylabel('Count')
plt.title('Distribution of Obesity Classes')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "01_class_distribution.png"))
plt.close()

# === 2) Boxplots for numerical features ===
numerical = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
for col in numerical:
    plt.figure(figsize=(8,4))
    data = [df[df['NObeyesdad']==cls][col] for cls in class_order]
    plt.boxplot(data, labels=class_order)
    plt.xticks(rotation=45)
    plt.xlabel('Obesity Class')
    plt.ylabel(col)
    plt.title(f'{col} by Obesity Class')
    plt.tight_layout()
    # save each as 02_box_<col>.png
    filename = f"02_box_{col.lower()}.png"
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

# === 3) Stacked bar charts for categorical features ===
categorical = [
    'Gender','family_history_with_overweight','FAVC',
    'CAEC','SMOKE','SCC','CALC','MTRANS'
]
for attr in categorical:
    cnt = df.groupby([attr, 'NObeyesdad']).size().unstack(fill_value=0)
    cnt = cnt[class_order]  # ensure consistent order
    fig = cnt.plot(
        kind='bar',
        stacked=True,
        figsize=(8,4)
    ).get_figure()
    plt.xlabel(attr)
    plt.ylabel('Count')
    plt.title(f'Distribution of Obesity Classes by {attr}')
    plt.legend(title='Class', bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    filename = f"03_stacked_{attr.lower()}.png"
    fig.savefig(os.path.join(outdir, filename))
    plt.close(fig)

print(f"All figures saved to ./{outdir}/")
