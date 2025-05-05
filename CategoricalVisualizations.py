import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Clean/recreate the output folder at the start of each run
import shutil

#Create output directory
output_dir = "CategoricalVisualizations_Outputs"
os.makedirs(output_dir, exist_ok=True)
 
#Deletes files if already exists and replaces with new ones 
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


#Load and clean dataset
df = pd.read_csv("CategoricalDataset.csv")
df = df.dropna(subset=["Height", "family_history_with_overweight", "FAF", "CH2O", "CALC", "NObeyesdad", "BMI", "FCVC", "TUE"])

#Map labels
df["family_history_with_overweight"] = df["family_history_with_overweight"].map({0: "No", 1: "Yes"})
df["CALC"] = df["CALC"].map({0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"})
df["CALC"] = pd.Categorical(df["CALC"], categories=["Always", "Frequently", "Sometimes", "No"], ordered=True)

sns.set(style="whitegrid")

#1. HISTOGRAM: Height Histogram by Family History
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="Height", hue="family_history_with_overweight", kde=True, multiple="stack")
plt.title("Height by Family History of Overweight")
plt.xlabel("Height (m)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig(f"{output_dir}/1_height_by_family_history.png")
plt.show()

#2.BOX PLOT: Physical Activity by Obesity Class
plt.figure(figsize=(8, 6))
sns.boxplot(x="NObeyesdad", y="FAF", data=df)
plt.title("Physical Activity Frequency by Obesity Class")
plt.xlabel("Obesity Class")
plt.ylabel("FAF (hrs)")
plt.tight_layout()
plt.savefig(f"{output_dir}/2_faf_by_obesity_class.png")
plt.show()

#3. HISTOGRAM: Water Intake by Obesity Class
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="CH2O", hue="NObeyesdad", multiple="stack", kde=True)
plt.title("Water Intake by Obesity Class")
plt.xlabel("Daily Water (liters)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig(f"{output_dir}/3_ch2o_by_obesity_class.png")
plt.show()

#4. STACKED BAR GRAPH: Alcohol Consumption by Obesity Class
alcohol_counts = df.groupby(["CALC", "NObeyesdad"]).size().unstack(fill_value=0).loc[["Always", "Frequently", "Sometimes", "No"]]
ax = alcohol_counts.plot(kind="bar", stacked=True, figsize=(10, 6))
ax.set_title("Alcohol Consumption by Obesity Class")
ax.set_xlabel("Alcohol Frequency")
ax.set_ylabel("Number of People")
ax.legend(title="Obesity Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/4_alcohol_by_obesity_class.png")
plt.show()

#5. CORRELATION MATRIX: Correlation Matrix between FMI, FCVC, FAF, CH20, TUE
plt.figure(figsize=(8, 6))
corr_columns = ["BMI", "FCVC", "FAF", "CH2O", "TUE"]
correlation = df[corr_columns].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: BMI & Lifestyle Factors")
plt.tight_layout()
plt.savefig(f"{output_dir}/5_correlation_matrix.png")
plt.show()
