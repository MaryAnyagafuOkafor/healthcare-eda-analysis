#!/usr/bin/env python
# coding: utf-8

# 1. IMPORT LIBRARIES

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

print("Dataset loaded successfully.")


# 2. DATA SET EXPLORATION AND DATA CLEANING

# In[5]:


# determine number of rows and columns of the data set

rows = len(df)
columns = len(df.columns)

print("Number of rows:", rows)
print("Number of columns:", columns)


# In[6]:


# display first records

print(df.head())


# In[7]:


# display last records

print(df.tail())


# 3. IDENTIFY ALL VARIABLES IN THE DATA SET AND THEIR DATA TYPES

# In[9]:


for column in df.columns:
    print(column, ":", df[column].dtype)


# 4. CHECK FOR ALL MISSING VALUES

# In[11]:


for column in df.columns:
    missing = 0
    
    for value in df[column]:
        if pd.isna(value):
            missing += 1
    
    print(column, ":", missing)


# 5. CHECK FOR DUPLICATES IN THE DATA SET

# In[13]:


# Display number of duplicate records

duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)


# In[14]:


# See duplicates 
duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

duplicates


# 6. CHECK FOR NUMERICAL ERRORS

# In[16]:


# check numerical columns

numerical_columns = ["Age", "Billing Amount", "Room Number"]

for column in numerical_columns:
    print("\n", column)
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())


# In[17]:


# check negative billing amounts

negative_billing = 0

for value in df["Billing Amount"]:
    if value < 0:
        negative_billing += 1

print("Negative billing values:", negative_billing)


# In[18]:


# Display negative billing amount

negative_billing = df[df["Billing Amount"] < 0]

print(negative_billing["Billing Amount"])


# 7. CHECK FOR CATEGORICAL VARIABLES ERRORS

# In[20]:


# check for unexpected categories/trailing spaces

categorical_columns = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results"
]

for column in categorical_columns:
    print("\n", column)
    print(df[column].unique())


# In[21]:


# check for spelling errors

print("\n", column)
print(df[column].value_counts())


# 8. CHECK FOR DATES ERROR

# In[23]:


print(df["Date of Admission"].dtype)
print(df["Discharge Date"].dtype)


# In[24]:


# convert dates

df["Date of Admission"] = pd.to_datetime(
    df["Date of Admission"],
    errors="coerce"
)

df["Discharge Date"] = pd.to_datetime(
    df["Discharge Date"],
    errors="coerce"
)


# 9. DESCRIPTIVE STATISTICS FOR NUMERICAL VARIABLES

# a. Measures of Location
# For variables such as Age and Billing Amount:
# - Mean
# - Median
# - Mode

# In[27]:


# Age

print("Mean:", df["Age"].mean())          # Mean

print("Median:", df["Age"].median())     # Median

print("Mode:", df["Age"].mode())       # Mode


# In[28]:


# Billing Amount

print("Mean:", df["Billing Amount"].mean())          # Mean

print("Median:", df["Billing Amount"].median())     # Median


# b. Measures of Variability
# Include:
# - Range
# - Variance
# - Standard deviation
# - IQR
# - Percentiles
# - Minimum
# - Maximum

# In[30]:


# Age

print("Range:", df["Age"].max() - df["Age"].min())     # Range  

print("Minimum:", df["Age"].min())       # minimum

print("Maximum:", df["Age"].max())       # maximum

print("Variance:", df["Age"].var())          # variance

print("Standard deviation:", df["Age"].std())     # standard deviation

print("IQR:", df["Age"].quantile(0.75) - df["Age"].quantile(0.25))    # Interquartile

print("Percentile Q1:", df["Age"].quantile(0.25))

print("Percentile Q2", df["Age"].quantile(0.75))

print("Skewness:", df["Age"].skew())       # skewness

print("Kurtosis:", df["Age"].kurt())      # kurtosis


# In[31]:


# Billing Amount

print("Range:", df["Billing Amount"].max() - df["Billing Amount"].min())     # Range  

print("Minimum:", df["Billing Amount"].min())       # minimum

print("Maximum:", df["Billing Amount"].max())       # maximum

print("Variance:", df["Billing Amount"].var())          # variance

print("Standard deviation:", df["Billing Amount"].std())     # standard deviation

print("IQR:", df["Billing Amount"].quantile(0.75) - df["Billing Amount"].quantile(0.25))    # Interquartile

print("Percentile Q1:", df["Billing Amount"].quantile(0.25))

print("Percentile Q2:", df["Billing Amount"].quantile(0.75))

print("Skewness:", df["Billing Amount"].skew())       # skewness

print("Kurtosis:", df["Billing Amount"].kurt())       # kurtosis


# c. Bivariate Analysis((Relationship Between Variables)

# In[33]:


# Age vs Billing Amount

print("cov_matrix:",  df[["Age", "Billing Amount"]].cov())     # covariance

from scipy.stats import pearsonr    
r, p = pearsonr(df["Age"], df["Billing Amount"])       # correlation
print(f"Pearson r = {r:.4f}")
print(f"p-value   = {p:.4g}")


# 10. DESCRIPTIVE STATISTICS FOR CATEGOTICAL VARIABLES

# In[35]:


# count all categorical variables

for column in categorical_columns:
    print("\n", column)
    print(df[column].value_counts())


# In[36]:


# percentage proportion of gender

df["Gender"].value_counts(normalize=True) * 100


# In[37]:


# mode(highest occurence) in category

for column in categorical_columns:
    print(column, ":", df[column].mode()[0])


# In[38]:


# count number of all categorical variables

for column in categorical_columns:
    print(column, ":", df[column].nunique())


# 11. VISUAL REPRESENTATION OF DESCRIPTIVE STATISTICS

# In[40]:


import os

os.makedirs("figures", exist_ok=True)


# 1. Visual Analysis of Numerical Variables

# In[42]:


# Age histogram with KDE (Univariate)

COLOR = sns.color_palette("colorblind")[2]

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Age", bins=20, color=COLOR, edgecolor="black")

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram + KDE Representing Age Distribution")
plt.tight_layout()
plt.show()


colors = sns.color_palette("colorblind")
COLOR = colors[2]

mean_age = df["Age"].mean()
median_age = df["Age"].median()
mode_age = df["Age"].mode()[0]

plt.figure(figsize=(9, 5))

# KDE
sns.kdeplot(
    df["Age"],
    color=COLOR,
    linewidth=2,
    alpha=0.8,
    fill=True
)

# Mean line
plt.axvline(
    mean_age,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_age:.2f}"
)

# Median line
plt.axvline(
    median_age,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Median = {median_age:.2f}"
)

# Mode line
plt.axvline(
    mode_age,
    color="purple",
    linestyle="--",
    linewidth=2,
    label=f"Mode = {mode_age:.2f}"
)

plt.xlabel("Age")
plt.ylabel("Density")
plt.title("KDE of Age Distribution with Mean and Median")

plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_1_Age_Distribution.png",
            dpi=300, bbox_inches="tight")


# In[43]:


# Age Spread (to show IQR, whiskers = range (excluding outliers), dots = outlier) (univariate)

COLOR=sns.color_palette("colorblind")[2]

plt.figure(figsize=(9, 4))
sns.boxplot(x=df["Age"], color=COLOR)
plt.xlabel("Age")
plt.title("Boxplot Representing Age — Spread")
plt.tight_layout()
plt.show()


print("Minimum:", df["Age"].min())
print("Maximum:", df["Age"].max())
print("Median:", df["Age"].median())        
print("25th percentile:", df["Age"].quantile(0.25))      # 25th percentile
print("75th percentile:", df["Age"].quantile(0.75))      # 75th percentile
print("IQR:", df["Age"].quantile(0.75) - df["Age"].quantile(0.25))    # Interquartile Range

plt.savefig("figures/Figure_2_Boxplot_Representing_Age_Spread.png",
            dpi=300, bbox_inches="tight")


# In[44]:


# Billing Amount histogram with KDE (Univariate)

COLOR = sns.color_palette("colorblind")[2]

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Billing Amount", bins=20, color=COLOR, edgecolor="black")

plt.xlabel("Billing Amount")
plt.ylabel("Frequency")
plt.title("Histogram + KDE Representing Billing Amount Distribution")
plt.tight_layout()
plt.show()


colors = sns.color_palette("colorblind")
COLOR = colors[2]

mean_age = df["Age"].mean()
median_age = df["Age"].median()

plt.figure(figsize=(9, 5))


colors = sns.color_palette("colorblind")
COLOR = colors[2]

mean_billing = df["Billing Amount"].mean()
median_billing = df["Billing Amount"].median()

plt.figure(figsize=(9, 5))

# KDE
sns.kdeplot(
    df["Billing Amount"],
    color=COLOR,
    linewidth=2,
    alpha=0.8,
    fill=True
)

# Mean line
plt.axvline(
    mean_billing,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_billing:,.2f}"
)

# Median line
plt.axvline(
    median_billing,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Median = {median_billing:,.2f}"
)

plt.xlabel("Billing Amount")
plt.ylabel("Density")
plt.title("KDE of Billing Amount Distribution with Mean and Median")

plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_3_Billing_Amount_Distribution.png",
            dpi=300, bbox_inches="tight")


# In[45]:


# Billing Amount Spread (to show IQR, whiskers = range (excluding outliers), dots = outlier) (univariate)

COLOR=sns.color_palette("colorblind")[2]

plt.figure(figsize=(9, 4))
sns.boxplot(x=df["Billing Amount"], color=COLOR)
plt.xlabel("Billing Amount")
plt.title("Boxplot Representing Billing Amount — Spread")
plt.tight_layout()
plt.show()


print("Minimum:", df["Billing Amount"].min())
print("Maximum:", df["Billing Amount"].max())
print("Median:", df["Billing Amount"].median())        
print("25th percentile:", df["Billing Amount"].quantile(0.25))      # 25th percentile
print("75th percentile:", df["Billing Amount"].quantile(0.75))      # 75th percentile
print("IQR:", df["Billing Amount"].quantile(0.75) - df["Billing Amount"].quantile(0.25))    # Interquartile Range

plt.savefig("figures/Figure_4_Boxplot_Representing_Billing_Amount_Spread.png",
            dpi=300, bbox_inches="tight")


# 2. Visual Analysis of Categorical Variable 

# In[47]:


# Gender distribution (Univariate)

COLOR = sns.color_palette("colorblind")[2]

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Gender",
    color=COLOR
)

total = len(df)

for p in ax.patches:
    count = int(p.get_height())
    percentage = count / total * 100

    # Count above the bar
    ax.annotate(
        f"{count:,}",
        (p.get_x() + p.get_width()/2, count),
        ha="center",
        va="bottom",
        fontsize=9
    )

    # Percentage inside the bar
    ax.annotate(
        f"{percentage:.1f}%",
        (p.get_x() + p.get_width()/2, count / 2),
        ha="center",
        va="center",
        fontsize=9,
        color="white"
    )

plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.title("Bar Chart Representing Gender Distribution")
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_5_Bar_Chart_Representing_Gender_Distribution.png",
            dpi=300, bbox_inches="tight")


# In[48]:


# Count each blood type (univariate)

blood_counts = df["Blood Type"].value_counts()

# Identify the most frequent blood type
lowest_blood_type = blood_counts.idxmin()

# Green for the highest, gray for the others
COLOR = sns.color_palette("colorblind")[2]

bar_colors = [
    COLOR if blood_type == lowest_blood_type else "lightgray"
    for blood_type in blood_counts.index
]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    blood_counts.index,
    blood_counts.values,
    color=bar_colors,
    edgecolor="black"
)

# Add frequency labels
for bar, value in zip(bars, blood_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Blood Type")
plt.ylabel("Number of Patients")
plt.title("Bar Plot Representing The Lowest Blood Type Distribution")
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_6_Barplot_Representing_Blood_Type_Distribution.png",
            dpi=300, bbox_inches="tight")


# 3. Relationships/Comparison 

# In[50]:


# distribution of gender across medical conditions (Bivariate)

gender_condition = pd.crosstab(
    df["Medical Condition"],
    df["Gender"]
)

# Colorblind-friendly colours
colors = sns.color_palette("colorblind")

# Create stacked bar chart
gender_condition.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[colors[2], colors[0]],
    edgecolor="black"
)

plt.xlabel("Medical Conditions")
plt.ylabel("Number of Patients")
plt.title("Bar Chart Representing Distribution of Medical Conditions By Gender")
plt.xticks(rotation=30)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


plt.savefig("figures/Figure_7_Bar_Chart_Representing_Distribution_of_Medical_Conditions_By_Gender.png",
            dpi=300, bbox_inches="tight")


# In[51]:


# visualizing age by medical condition (Bivariate)

COLOR = sns.color_palette("colorblind")[2]

average_age = df.groupby("Medical Condition")["Age"].mean()

plt.figure(figsize=(10, 6))

bars = plt.bar(
    average_age.index,
    average_age.values,
    color=COLOR,
    edgecolor="black"
)

for bar, value in zip(bars, average_age.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Medical Condition")
plt.ylabel("Average Age")
plt.title("Bar Chart Representing Average Age By Medical Conditions")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# boxplot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Age",
    color=sns.color_palette("colorblind")[2]
)

plt.xlabel("Medical Condition")
plt.ylabel("Age")
plt.title("Boxplot Representing Age Distribution by Medical Condition")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_8_Bar Chart_Boxplot_Representing_Average_Age_By_Medical_Conditions.png",
            dpi=300, bbox_inches="tight")


# In[52]:


# visualizing billing amounts by medical condition (Bivariate)

COLOR = sns.color_palette("colorblind")[2]

# Calculate average billing amount for each medical condition
average_billing = df.groupby("Medical Condition")["Billing Amount"].mean()

# Identify the smallest medical condition
smallest_condition = average_billing.idxmin()
smallest_value = average_billing.min()

# Make all bars gray except the smallest
bar_colors = [
    COLOR if condition == smallest_condition else "lightgray"
    for condition in average_billing.index
]

# Create bar chart
plt.figure(figsize=(10, 6))

bars = plt.bar(
    average_billing.index,
    average_billing.values,
    color=bar_colors,
    edgecolor="black"
)

# Display the value on every bar
for bar, value in zip(bars, average_billing.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )



plt.xlabel("Medical Conditions")
plt.ylabel("Average Billing Amount")
plt.title("Bar Chart Representing Average Billing Amount By Medical Conditions")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# box plot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Billing Amount",
    color=sns.color_palette("colorblind")[2]
)

plt.xlabel("Medical Condition")
plt.ylabel("Billing Amount")
plt.title("Boxplot Representing Billing Amount Distribution by Medical Condition")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_9_Bar_chart_Boxplot_Representing_Average_Billing_Amount_By_Medical_Conditions.png",
            dpi=300, bbox_inches="tight")


# In[53]:


# Stacked histogram showing Billing Amount distribution by Admission Type (Bivariate)

sns.histplot(
    data=df,
    x="Billing Amount",
    bins=20,
    hue="Admission Type",
    multiple="stack",
    palette="colorblind",
    edgecolor="black"
)

plt.xlabel("Billing Amount")
plt.ylabel("Number of Patients")
plt.title("Stacked Histogram Representing The Distribution of Billing Amount by Admission Type")
plt.tight_layout()
plt.show()


# boxplot

COLOR = sns.color_palette("colorblind")[2]

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Admission Type", y="Billing Amount",color=COLOR)
plt.xlabel("Admission Type")
plt.ylabel("Billing Amount")
plt.title("Boxplot Representing Billing Amount by Admission Type")
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_10_Stacked_Histogram_Boxplot_Representing_Billing_Amount_by_Admission_Type.png",
            dpi=300, bbox_inches="tight")


# 4. COVARIANCE

# In[55]:


# Age vs Billing Amount

print("cov_matrix:",  df[["Age", "Billing Amount"]].cov())     # covariance


# 5. CORRELATION

# In[57]:


# Correlation of numerical variables for Age and Billing Amount (Bivariate)

from scipy.stats import pearsonr

# Select the two numerical variables
data = df[["Age", "Billing Amount"]]

# Calculate Pearson correlation and p-value
r, p = pearsonr(data["Age"], data["Billing Amount"])

# Colour settings
COLOR = sns.color_palette("colorblind")[2]
DARK_GREEN = sns.dark_palette(COLOR, reverse=False, as_cmap=True)


# --------------------------------------------------
# Regression Plot
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.regplot(
    data=df,
    x="Age",
    y="Billing Amount",
    scatter_kws={"alpha": 0.3, "s": 10, "color": COLOR},
    line_kws={"color": "red", "lw": 2}
)

plt.xlabel("Age")
plt.ylabel("Billing Amount")

plt.title(
    f"Regression Plot Representing Age vs Billing Amount of the Healthcare Dataset\n"
    f"Pearson r = {r:.3f}, p-value = {p:.3g}"
)

plt.tight_layout()

plt.savefig(
    "figures/Figure_11_Regression_Plot_Age_Billing_Amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# Pearson Correlation Heatmap
# --------------------------------------------------

corr = data.corr(method="pearson")

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap=DARK_GREEN,
    center=0,
    fmt=".3f",
    square=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Pearson Correlation"}
)

plt.title(
    f"Pearson Correlation Representing Age vs Billing Amount of the Healthcare Dataset\n"
    f"r = {r:.3f}, p-value = {p:.3g}"
)

plt.tight_layout()

plt.savefig(
    "figures/Figure_11_Pearson_Correlation_Age_Billing_Amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[58]:


# comparing distribution against groups

ct = pd.crosstab(
    df["Admission Type"],
    df["Medical Condition"])

ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

ct_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 6),
    color=sns.color_palette("colorblind", n_colors=ct.shape[1])
)

plt.xlabel("Admission Type")
plt.ylabel("Percentage (%)")
plt.title("Stacked Bar Plot Representing Proportion of Medical Conditions by Admission Type")
plt.xticks(rotation=0)
plt.legend(title="Medical Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.savefig("figures/Figure_12_Stacked_Barplot_Representing_Proportion_of_Medical_Coditions_by_Admission_Type.png",
            dpi=300, bbox_inches="tight")


# In[59]:


# Extract year from Date of Admission

df["Year"] = df["Date of Admission"].dt.year

# Count patients per year
year_counts = df["Year"].value_counts().sort_index()

plt.figure(figsize=(9, 5))

plt.plot(
    year_counts.index,
    year_counts.values,
    marker="o",
    color=sns.color_palette("colorblind")[2],
    linewidth=1
)

# Add values to each point
for year, value in zip(year_counts.index, year_counts.values):
    plt.text(
        year,
        value,
        f"{value:,}",
        ha="left",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Year")
plt.ylabel("Number of Patients")
plt.title("Line Plot Representing Number of Patients by Year")

plt.tight_layout()
plt.show()
plt.savefig("figures/Figure_13_Line_plot_Representing_Number_of-Patients_Per-Year.png",
            dpi=300, bbox_inches="tight")

