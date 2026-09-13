#!/usr/bin/env python
# coding: utf-8

# # Healthcare Dataset Analysis
# 
# # This notebook performs Exploratory Data Analysis (EDA) and Visualization on the healthcare dataset.

# In[3]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# In[4]:


# Load the Dataset
# Read the CSV file

df = pd.read_csv('healthcare_dataset_cleaned.csv')
print(f"Dataset loaded! Shape: {df.shape}")


# In[5]:


# Display first few rows to understand the data

print("\nFirst 5 rows of the dataset:")
print(df.head())


# # Data Cleaning

# In[7]:


# Data Cleaning 

# View actual null values as they appear in the dataset
print("=== ACTUAL NULL VALUES IN THE DATASET ===\n")

# Find all rows with null values
null_rows = df[df.isnull().any(axis=1)]

if len(null_rows) > 0:
    print(f"Found {len(null_rows)} rows with null values\n")
    
    # Display the actual rows with nulls
    print("Actual rows with null values:")
    print(null_rows)
else:
    print("✅ No null values found in the dataset")


# In[8]:


# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values (similar to handling NaN in the course example)
df_clean = df.dropna()
print(f"\nRows after dropping null values: {df_clean.shape[0]} (dropped {df.shape[0] - df_clean.shape[0]} rows)")


# In[9]:


# Check rows before and after removing missing values
print("=== ROW COUNT COMPARISON ===\n")

# Before dropping
print(f"Rows before removing missing values: {df.shape[0]}")

# After dropping
df_clean = df.dropna()
print(f"Rows after removing missing values: {df_clean.shape[0]}")

# Calculate difference
rows_dropped = df.shape[0] - df_clean.shape[0]
print(f"Total rows dropped: {rows_dropped}")

# Show percentage dropped
if df.shape[0] > 0:
    percentage_dropped = (rows_dropped / df.shape[0]) * 100
    print(f"Percentage of rows dropped: {percentage_dropped:.2f}%")


# In[10]:


# Convert date columns to datetime for analysis

df_clean['Date of Admission'] = pd.to_datetime(df_clean['Date of Admission'])
df_clean['Discharge Date'] = pd.to_datetime(df_clean['Discharge Date'])


# In[11]:


# Most concise way to convert negative to positive
df_clean['Billing Amount'] = df_clean['Billing Amount'].abs()


# In[12]:


# Check how many negative values exist
negative_bills = df_clean[df_clean['Billing Amount'] < 0]
print(f"Number of negative bills: {len(negative_bills)}")
print(f"Percentage: {len(negative_bills)/len(df_clean)*100:.2f}%")

# View the negative values
print(negative_bills[['Billing Amount', 'Medical Condition', 'Admission Type']].head(10))


# # Location and Variability

# In[14]:


#------- Mean Calculation (Manual) -------

# Following the course example for mean calculation
x = df_clean['Billing Amount'].values  # list with data values
n = len(x)  # number of data values of x
my_sum = sum(x)  # sum all values of x up
my_mean = my_sum / n  # get mean of x
print(f"\n--- Manual Mean Calculation for Billing Amount ---")
print(f"Manual Mean: {my_mean:.2f}")



# In[15]:


#------- Mean Calculation (Using statistics library) -------

import statistics
my_mean = statistics.mean(x)  # get mean of x
print(f"Statistics library Mean: {my_mean:.2f}")


# In[16]:


#------- Median Calculation (Manual) -------

x = df_clean['Billing Amount'].values
n = len(x)
x_sorted = sorted(x)  # sort data
if n % 2 == 0:  # even number of values
    median1 = x_sorted[n//2]
    median2 = x_sorted[n//2 - 1]
    median = (median1 + median2) / 2
else:  # odd number of values
    median = x_sorted[n//2]
print(f"\n--- Manual Median Calculation for Billing Amount ---")
print(f"Manual Median: {median:.2f}")


# In[17]:


#------- Median Calculation (Using statistics library) -------

median = statistics.median(x)
print(f"Statistics library Median: {median:.2f}")


# In[18]:


#------- Mode Calculation (Manual) -------

from collections import Counter
x = df_clean['Billing Amount'].values
n = len(x)
frequencies = Counter(x)  # count frequencies of values
modes_dict = dict(frequencies)  # store frequencies in dictionary


# calculated modes
modes = [k for k, v in modes_dict.items() if v == max(list(frequencies.values()))]
print(f"\n--- Manual Mode Calculation for Billing Amount ---")
print(f"Manual Modes: {modes[:5]}... (showing first 5)")  # There might be many modes
print(f"Number of unique modes: {len(modes)}")



# In[19]:


try:
    mode_value = statistics.mode(x)
    print(f"Statistics library Mode: {mode_value:.2f}")
except statistics.StatisticsError:
    print("Statistics library Mode: Multiple modes found or no unique mode.")


# In[20]:


#------- Variance Calculation (Manual) -------

x = df_clean['Billing Amount'].values
n = len(x)
mean_val = sum(x) / n
my_var = sum((xi - mean_val) ** 2 for xi in x) / n
print(f"\n--- Manual Variance Calculation for Billing Amount ---")
print(f"Manual Variance: {my_var:.2f}")


# In[21]:


#------- Variance Calculation (Using statistics library) -------

my_var = statistics.variance(x)  # Sample variance
print(f"Statistics library Variance (Sample): {my_var:.2f}")


# In[22]:


#------- Standard Deviation Calculation (Manual) -------

my_std = my_var ** 0.5
print(f"\n--- Manual Standard Deviation Calculation for Billing Amount ---")
print(f"Manual Standard Deviation: {my_std:.2f}")


# In[23]:


#------- Standard Deviation Calculation (Using statistics library) -------

my_std = statistics.stdev(x)  # Sample standard deviation
print(f"Statistics library Standard Deviation (Sample): {my_std:.2f}")


# # Quartiles, IQR, Skewness, and Kurtosis

# In[25]:


#------- Quartiles Calculation (Q1, Q2/Median, Q3) -------
x = df_clean['Billing Amount'].values
sorted_x = np.sort(x)
n = len(sorted_x)

# Q1 = 25th percentile
q1 = np.percentile(sorted_x, 25)
# Q2 = 50th percentile (Median)
q2 = np.percentile(sorted_x, 50)
# Q3 = 75th percentile
q3 = np.percentile(sorted_x, 75)

print(f"\n--- Quartiles for Billing Amount ---")
print(f"Q1 (25th Percentile): ${q1:,.2f}")
print(f"Q2 (50th Percentile / Median): ${q2:,.2f}")
print(f"Q3 (75th Percentile): ${q3:,.2f}")


# In[26]:


#------- Interquartile Range (IQR) Calculation -------
iqr = q3 - q1
print(f"\n--- Interquartile Range (IQR) for Billing Amount ---")
print(f"IQR (Q3 - Q1): ${iqr:,.2f}")

# Calculate lower and upper bounds for outlier detection
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Lower Bound for Outliers: ${lower_bound:,.2f}")
print(f"Upper Bound for Outliers: ${upper_bound:,.2f}")

# Count outliers
outliers = [val for val in x if val < lower_bound or val > upper_bound]
print(f"Number of Outliers: {len(outliers)}")
print(f"Percentage of Outliers: {(len(outliers)/len(x))*100:.2f}%")


# In[27]:


#------- Skewness Calculation -------

# Skewness measures asymmetry of the distribution
# Positive skew = tail on right, Negative skew = tail on left
mean_val = np.mean(x)
median_val = np.median(x)
std_val = np.std(x)
n = len(x)

# Calculate skewness using the formula
skewness = sum((xi - mean_val) ** 3 for xi in x) / (n * std_val ** 3)
print(f"\n--- Skewness for Billing Amount ---")
print(f"Skewness: {skewness:.4f}")
if skewness > 0.5:
    print("Interpretation: Positive skew (right-skewed) - Tail extends to the right")
    print("              Most bills are low, but a few are very expensive")
elif skewness < -0.5:
    print("Interpretation: Negative skew (left-skewed) - Tail extends to the left")
    print("              Most bills are high, but a few are very cheap")
else:
    print("Interpretation: Approximately symmetric distribution")

# Using scipy for comparison (if available)
try:
    from scipy.stats import skew
    scipy_skew = skew(x)
    print(f"Scipy Skewness: {scipy_skew:.4f}")
except:
    print("Scipy not available for skewness check")


# In[28]:


#------- Kurtosis Calculation -------

# Kurtosis measures the "tailedness" of the distribution
# High kurtosis = heavy tails (more outliers)
# Low kurtosis = light tails (fewer outliers)
kurtosis = sum((xi - mean_val) ** 4 for xi in x) / (n * std_val ** 4) - 3
print(f"\n--- Kurtosis for Billing Amount ---")
print(f"Kurtosis (Excess): {kurtosis:.4f}")
if kurtosis > 1:
    print("Interpretation: Leptokurtic (heavy tails) - More outliers than normal distribution")
    print("              There are extreme values far from the mean")
elif kurtosis < -1:
    print("Interpretation: Platykurtic (light tails) - Fewer outliers than normal distribution")
    print("              Values are concentrated around the mean")
else:
    print("Interpretation: Mesokurtic - Similar to normal distribution")

# Using scipy for comparison (if available)
try:
    from scipy.stats import kurtosis as kurt
    scipy_kurt = kurt(x)
    print(f"Scipy Kurtosis: {scipy_kurt:.4f}")
except:
    print("Scipy not available for kurtosis check")


# # Location and Variability for Numerical Columns
# 

# In[30]:


#------- Calculate for all numerical columns -------

numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
print("\nNumerical columns in dataset:")
print(list(numerical_cols))

# Create a summary DataFrame
summary_data = []
for col in numerical_cols:
    data = df_clean[col].values
    n = len(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    try:
        mode_val = statistics.mode(data)
    except:
        mode_val = np.nan

    # Quartiles
    q1_val = np.percentile(data, 25)
    q2_val = np.percentile(data, 50)
    q3_val = np.percentile(data, 75)
    iqr_val = q3_val - q1_val
    
    # Skewness and Kurtosis
    std_val = np.std(data)
    skew_val = sum((xi - mean_val) ** 3 for xi in data) / (n * std_val ** 3) if std_val > 0 else 0
    kurt_val = sum((xi - mean_val) ** 4 for xi in data) / (n * std_val ** 4) - 3 if std_val > 0 else 0

    
    variance_val = np.var(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    summary_data.append({
        'Column': col,
        'Count': n,
        'Mean': mean_val,
        'Median': median_val,
        'Mode': mode_val,
        'Q1': q1_val,
        'Q2 (Median)': q2_val,
        'Q3': q3_val,
        'IQR': iqr_val,
        'Variance': variance_val,
        'Std Dev': std_val,
        'Skewness': skew_val,
        'Kurtosis': kurt_val,
        'Min': min_val,
        'Max': max_val
    })

summary_df = pd.DataFrame(summary_data)
print("\n--- Summary Statistics for Numerical Columns ---")
print(summary_df.round(2))


# # Data Visualization

# In[32]:


#------- Import Visualization Libraries -------

import seaborn as sns
import pandas as pd
import numpy as np


# In[33]:


#------- Load data (already loaded above) -------


# # Distribution of Billing Amount

# In[35]:


#------- Histogram of Billing Amount with Statistics -------

plt.figure(figsize=(12, 7))

# Create histogram with KDE
ax = sns.histplot(df_clean['Billing Amount'], bins=30, kde=True, color='steelblue', alpha=0.7)

# Add vertical lines for mean, median, and quartiles
mean_val = np.mean(df_clean['Billing Amount'])
median_val = np.median(df_clean['Billing Amount'])
q1_val = np.percentile(df_clean['Billing Amount'], 25)
q3_val = np.percentile(df_clean['Billing Amount'], 75)

plt.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_val:,.0f}')
plt.axvline(q1_val, color='orange', linestyle=':', linewidth=2, label=f'Q1: ${q1_val:,.0f}')
plt.axvline(q3_val, color='purple', linestyle=':', linewidth=2, label=f'Q3: ${q3_val:,.0f}')

# Format x-axis as currency
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Add legend and labels
plt.title('Distribution of Billing Amount with Statistical Measures', fontsize=14, fontweight='bold')
plt.xlabel('Billing Amount', fontsize=12)
plt.ylabel('Number of Patients (Frequency)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, None)

plt.tight_layout()
plt.show()


# In[36]:


#------- Boxplot of Billing Amount with Statistics Displayed -------

plt.figure(figsize=(10, 8))
ax = sns.boxplot(y=df_clean['Billing Amount'], color='lightblue', width=0.5)

# Format y-axis as currency
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Calculate statistics
data = df_clean['Billing Amount'].values
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_whisker = np.min(data[data >= q1 - 1.5 * iqr])
upper_whisker = np.max(data[data <= q3 + 1.5 * iqr])
mean_val = np.mean(data)

# Count outliers
outliers = [val for val in data if val < q1 - 1.5 * iqr or val > q3 + 1.5 * iqr]
num_outliers = len(outliers)
outlier_min = min(outliers) if outliers else None
outlier_max = max(outliers) if outliers else None

# Add horizontal lines for statistics
plt.axhline(q1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(q2, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(q3, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(mean_val, color='red', linestyle='-.', linewidth=1.5, alpha=0.7)

# Add text annotations on the right side of the plot
x_pos = 0.65  # Position on x-axis for text labels

# Q1 Annotation
plt.text(x_pos, q1, f'  Q1 = ${q1:,.0f}', 
         verticalalignment='center', fontsize=10, color='orange', fontweight='bold')

# Q2 (Median) Annotation
plt.text(x_pos, q2, f'  Median (Q2) = ${q2:,.0f}', 
         verticalalignment='bottom', fontsize=10, color='green', fontweight='bold')

# Q3 Annotation
plt.text(x_pos, q3, f'  Q3 = ${q3:,.0f}', 
         verticalalignment='center', fontsize=10, color='purple', fontweight='bold')

# Mean Annotation
plt.text(x_pos, mean_val, f'  Mean = ${mean_val:,.0f}', 
         verticalalignment='top', fontsize=10, color='red', fontweight='bold')

# IQR Annotation (placed in the middle of the box)
iqr_mid = (q1 + q3) / 2
plt.text(0.3, iqr_mid, f'IQR = ${iqr:,.0f}', 
         verticalalignment='center', horizontalalignment='center',
         fontsize=11, color='darkblue', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add title and labels
plt.title('Distribution of Billing Amount with Statistics', fontsize=14, fontweight='bold')
plt.ylabel('Billing Amount', fontsize=12)
plt.grid(True, alpha=0.3)

# Remove y-axis label to avoid clutter (optional)
plt.tight_layout()
plt.show()


# In[37]:


#------- Violin Plot of Billing Amount with Statistics Displayed -------

plt.figure(figsize=(10, 8))
ax = sns.violinplot(y=df_clean['Billing Amount'], color='steelblue', alpha=0.7, inner=None)

# Add boxplot inside violin for better statistics visualization
sns.boxplot(y=df_clean['Billing Amount'], color='white', width=0.15, ax=ax, 
            boxprops={'alpha': 0.5}, whiskerprops={'color': 'black'})

# Format y-axis as currency
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Calculate statistics
data = df_clean['Billing Amount'].values
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)
iqr = q3 - q1
mean_val = np.mean(data)
std_val = np.std(data)
min_val = np.min(data)
max_val = np.max(data)

# Calculate whiskers and outliers
lower_whisker = np.min(data[data >= q1 - 1.5 * iqr])
upper_whisker = np.max(data[data <= q3 + 1.5 * iqr])
outliers = [val for val in data if val < q1 - 1.5 * iqr or val > q3 + 1.5 * iqr]
num_outliers = len(outliers)

# Add horizontal lines for statistics
plt.axhline(q1, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axhline(q2, color='green', linestyle='--', linewidth=2, alpha=0.8)
plt.axhline(q3, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axhline(mean_val, color='red', linestyle='-.', linewidth=2, alpha=0.8)

# Add text annotations on the right side of the plot
x_pos = 0.65  # Position on x-axis for text labels

# Q1 Annotation
plt.text(x_pos, q1, f'  Q1 = ${q1:,.0f}', 
         verticalalignment='center', fontsize=10, color='orange', fontweight='bold')

# Q2 (Median) Annotation
plt.text(x_pos, q2, f'  Median (Q2) = ${q2:,.0f}', 
         verticalalignment='bottom', fontsize=10, color='green', fontweight='bold')

# Q3 Annotation
plt.text(x_pos, q3, f'  Q3 = ${q3:,.0f}', 
         verticalalignment='center', fontsize=10, color='purple', fontweight='bold')

# Mean Annotation
plt.text(x_pos, mean_val, f'  Mean = ${mean_val:,.0f}', 
         verticalalignment='top', fontsize=10, color='red', fontweight='bold')

# IQR Annotation (placed inside the violin at the middle of the box)
iqr_mid = (q1 + q3) / 2
plt.text(0.3, iqr_mid, f'IQR = ${iqr:,.0f}', 
         verticalalignment='center', horizontalalignment='center',
         fontsize=11, color='darkblue', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# Add title and labels
plt.title('Violin Plot of Billing Amount with Statistics', fontsize=14, fontweight='bold')
plt.ylabel('Billing Amount', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# # Distribution of Age

# In[39]:


# %% ------- Histogram of Age -------
plt.figure(figsize=(10, 6))
ax = sns.histplot(df_clean['Age'], bins=20, kde=True, color='teal', alpha=0.7)

# Add vertical lines for mean and median
mean_age = np.mean(df_clean['Age'])
median_age = np.median(df_clean['Age'])
plt.axvline(mean_age, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_age:.1f}')
plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f}')

plt.title('Patients Age Distribution Histogram', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Patients (Frequency)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, None)
plt.tight_layout()
plt.show()


# In[40]:


#------- Boxplot of Age by Gender with Statistics Displayed -------

plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Gender', y='Age', data=df_clean, palette='Set2')

# Calculate statistics for each gender
genders = df_clean['Gender'].unique()
colors = ['orange', 'green', 'purple', 'red']
stats_dict = {}

for i, gender in enumerate(genders):
    data = df_clean[df_clean['Gender'] == gender]['Age'].values
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mean_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    std_val = np.std(data)
    
    # Calculate whiskers and outliers
    lower_whisker = np.min(data[data >= q1 - 1.5 * iqr]) if any(data >= q1 - 1.5 * iqr) else q1
    upper_whisker = np.max(data[data <= q3 + 1.5 * iqr]) if any(data <= q3 + 1.5 * iqr) else q3
    outliers = [val for val in data if val < q1 - 1.5 * iqr or val > q3 + 1.5 * iqr]
    num_outliers = len(outliers)
    
    stats_dict[gender] = {
        'q1': q1, 'q2': q2, 'q3': q3, 'iqr': iqr,
        'mean': mean_val, 'min': min_val, 'max': max_val,
        'std': std_val, 'lower_whisker': lower_whisker,
        'upper_whisker': upper_whisker,
        'outliers': outliers, 'num_outliers': num_outliers
    }
    
    # Add horizontal lines for each gender at their respective x position
    x_pos = i  # 0 for first gender, 1 for second, etc.
    
    # Add mean line (red dashed)
    plt.axhline(y=mean_val, xmin=(x_pos-0.2)/len(genders), xmax=(x_pos+0.2)/len(genders),
                color='red', linestyle='-.', linewidth=2, alpha=0.8)
    
    # Add Q1 line (orange dotted)
    plt.axhline(y=q1, xmin=(x_pos-0.2)/len(genders), xmax=(x_pos+0.2)/len(genders),
                color='orange', linestyle=':', linewidth=2, alpha=0.8)
    
    # Add Q3 line (purple dotted)
    plt.axhline(y=q3, xmin=(x_pos-0.2)/len(genders), xmax=(x_pos+0.2)/len(genders),
                color='purple', linestyle=':', linewidth=2, alpha=0.8)

# Add title and labels
plt.title('Age Distribution by Gender with Statistics Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Age (Years)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistics text box for each gender
for i, gender in enumerate(genders):
    stats = stats_dict[gender]
    
    # Position the text box to the right of each boxplot
    x_pos = i + 0.15
    
    stats_text = f"""{gender}:
━━━━━━━━━━━━━━━━
Q1:  {stats['q1']:.1f}
Q2:  {stats['q2']:.1f} (Median)
Q3:  {stats['q3']:.1f}
IQR: {stats['iqr']:.1f}
Mean: {stats['mean']:.1f}
Std:  {stats['std']:.1f}
Outliers: {stats['num_outliers']}"""

    plt.text(x_pos + 0.25, stats['q2'], stats_text,
             fontsize=8, 
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85, edgecolor='gray'),
             family='monospace')

# Add legend for the lines
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='none', edgecolor='orange', linestyle=':', label='Q1 (25th Percentile)'),
    Patch(facecolor='none', edgecolor='green', linestyle='-', label='Q2 (Median)'),
    Patch(facecolor='none', edgecolor='purple', linestyle=':', label='Q3 (75th Percentile)'),
    Patch(facecolor='none', edgecolor='red', linestyle='-.', label='Mean')
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)

plt.tight_layout()
plt.show()

# Print statistics in console
print("\n" + "="*70)
print("AGE STATISTICS BY GENDER")
print("="*70)

for gender in genders:
    stats = stats_dict[gender]
    print(f"\n📊 {gender.upper()}:")
    print(f"  • Sample Size (n):     {len(df_clean[df_clean['Gender'] == gender]['Age']):,}")
    print(f"  • Minimum Age:         {stats['min']:.1f}")
    print(f"  • Maximum Age:         {stats['max']:.1f}")
    print(f"  • Mean Age:            {stats['mean']:.1f}")
    print(f"  • Q1 (25th %ile):      {stats['q1']:.1f}")
    print(f"  • Q2 (Median - 50th):  {stats['q2']:.1f}")
    print(f"  • Q3 (75th %ile):      {stats['q3']:.1f}")
    print(f"  • IQR (Q3 - Q1):       {stats['iqr']:.1f}")
    print(f"  • Standard Deviation:  {stats['std']:.1f}")
    print(f"  • Lower Whisker:       {stats['lower_whisker']:.1f}")
    print(f"  • Upper Whisker:       {stats['upper_whisker']:.1f}")
    print(f"  • Number of Outliers:  {stats['num_outliers']}")
    if stats['outliers']:
        outlier_str = ', '.join([f"{o:.1f}" for o in stats['outliers'][:5]])
        if len(stats['outliers']) > 5:
            outlier_str += f" ... and {len(stats['outliers']) - 5} more"
        print(f"  • Outlier Ages:        {outlier_str}")

print("\n" + "="*70)



# # Categorical Variable Analysis

# In[42]:


#------- Count Plot of Medical Condition -------

plt.figure(figsize=(12, 6))
ax = sns.countplot(y='Medical Condition', data=df_clean, 
                   order=df_clean['Medical Condition'].value_counts().index,
                   palette='Set3')

# Add count labels on bars
for i, bar in enumerate(ax.patches):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', ha='left', va='center', fontsize=10)

plt.title('Number of Patients with Medical Conditions', fontsize=14, fontweight='bold')
plt.ylabel('Medical Condition', fontsize=12)
plt.xlabel('', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[43]:


#------- Count Plot of Admission Type -------

plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Admission Type', data=df_clean, palette='viridis')

# Add count labels on bars
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.title('Number of Patients Admission Types', fontsize=14, fontweight='bold')
plt.ylabel('Patients', fontsize=12)
plt.xlabel('', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[44]:


#------- Count Plot of Test Results -------
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Test Results', data=df_clean, palette='coolwarm')

# Add count labels on bars
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.title('Number of Patients Test Results', fontsize=14, fontweight='bold')
plt.xlabel('', fontsize=12)
plt.ylabel('Patients', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# # Relationship Analysis

# In[46]:


#------- Billing Amount by Medical Condition (Clean Version with Table) -------

plt.figure(figsize=(14, 7))
ax = sns.boxplot(x='Medical Condition', y='Billing Amount', data=df_clean, palette='Set2')

# Format y-axis as currency
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Calculate statistics for each medical condition
conditions = df_clean['Medical Condition'].unique()
stats_summary = []

for condition in conditions:
    data = df_clean[df_clean['Medical Condition'] == condition]['Billing Amount'].values
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mean_val = np.mean(data)
    count = len(data)
    
    stats_summary.append({
        'Condition': condition,
        'n': count,
        'Q1': q1, 
        'Q2': q2, 
        'Q3': q3, 
        'IQR': iqr, 
        'Mean': mean_val
    })

# Add title and labels
plt.title('Patients Billing Amount by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Medical Condition', fontsize=12)
plt.ylabel('Billing Amount', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Create a statistics table at the bottom
stats_table = "Condition           |   n   |    Q1    |  Median (Q2)  |    Q3    |    IQR   |   Mean\n"
stats_table += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
for stat in stats_summary:
    # Truncate condition name if too long
    cond_name = stat['Condition'][:18] if len(stat['Condition']) > 18 else stat['Condition']
    stats_table += f"{cond_name:<18} | {stat['n']:5d} | ${stat['Q1']:8,.0f} | ${stat['Q2']:10,.0f}   | ${stat['Q3']:8,.0f} | ${stat['IQR']:8,.0f} | ${stat['Mean']:8,.0f}\n"

# Add text box with statistics
plt.text(0.02, 0.02, stats_table, 
         transform=plt.gca().transAxes, 
         fontsize=8, 
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
         family='monospace')

plt.tight_layout()
plt.show()

# Print statistics in console
print("\n" + "="*85)
print("BILLING AMOUNT STATISTICS BY MEDICAL CONDITION")
print("="*85)
print(f"{'Condition':<20} {'n':<6} {'Q1':<12} {'Median':<14} {'Q3':<12} {'IQR':<12} {'Mean':<12}")
print("-"*85)
for stat in stats_summary:
    print(f"{stat['Condition']:<20} {stat['n']:<6} ${stat['Q1']:>9,.0f}  ${stat['Q2']:>10,.0f}   ${stat['Q3']:>9,.0f}  ${stat['IQR']:>9,.0f}  ${stat['Mean']:>9,.0f}")
print("="*85)

# Find key insights
highest_mean = max(stats_summary, key=lambda x: x['Mean'])
lowest_mean = min(stats_summary, key=lambda x: x['Mean'])
highest_median = max(stats_summary, key=lambda x: x['Q2'])
lowest_median = min(stats_summary, key=lambda x: x['Q2'])

print("\n📊 KEY INSIGHTS:")
print(f"  • Highest Average Bill:   {highest_mean['Condition']} (${highest_mean['Mean']:,.2f})")
print(f"  • Lowest Average Bill:    {lowest_mean['Condition']} (${lowest_mean['Mean']:,.2f})")
print(f"  • Highest Median Bill:    {highest_median['Condition']} (${highest_median['Q2']:,.2f})")
print(f"  • Lowest Median Bill:     {lowest_median['Condition']} (${lowest_median['Q2']:,.2f})")

# Calculate the difference between highest and lowest
mean_diff = highest_mean['Mean'] - lowest_mean['Mean']
print(f"  • Difference (Highest - Lowest Mean): ${mean_diff:,.2f}")
print("="*85)


# In[47]:


#------- Billing Amount by Admission Type (Clean Version with Table) -------

plt.figure(figsize=(12, 7))
ax = sns.boxplot(x='Admission Type', y='Billing Amount', data=df_clean, palette='Set3')

# Format y-axis as currency
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Calculate statistics for each admission type
admission_types = df_clean['Admission Type'].unique()
stats_summary = []

for admission_type in admission_types:
    data = df_clean[df_clean['Admission Type'] == admission_type]['Billing Amount'].values
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mean_val = np.mean(data)
    count = len(data)
    
    stats_summary.append({
        'Admission Type': admission_type,
        'n': count,
        'Q1': q1, 
        'Q2': q2, 
        'Q3': q3, 
        'IQR': iqr, 
        'Mean': mean_val
    })

# Add title and labels
plt.title('Patients Billing Amount by Admission Type', fontsize=14, fontweight='bold')
plt.xlabel('', fontsize=12)
plt.ylabel('Billing Amount', fontsize=12)
plt.grid(True, alpha=0.3)

# Create a statistics table at the bottom
stats_table = "Admission Type     |   n   |    Q1    |  Median (Q2)  |    Q3    |    IQR   |   Mean\n"
stats_table += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
for stat in stats_summary:
    # Truncate admission type name if too long
    type_name = stat['Admission Type'][:18] if len(stat['Admission Type']) > 18 else stat['Admission Type']
    stats_table += f"{type_name:<18} | {stat['n']:5d} | ${stat['Q1']:8,.0f} | ${stat['Q2']:10,.0f}   | ${stat['Q3']:8,.0f} | ${stat['IQR']:8,.0f} | ${stat['Mean']:8,.0f}\n"

# Add text box with statistics
plt.text(0.02, 0.02, stats_table, 
         transform=plt.gca().transAxes, 
         fontsize=9, 
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
         family='monospace')

plt.tight_layout()
plt.show()

# Print statistics in console
print("\n" + "="*85)
print("BILLING AMOUNT STATISTICS BY ADMISSION TYPE")
print("="*85)
print(f"{'Admission Type':<18} {'n':<6} {'Q1':<12} {'Median':<14} {'Q3':<12} {'IQR':<12} {'Mean':<12}")
print("-"*85)
for stat in stats_summary:
    print(f"{stat['Admission Type']:<18} {stat['n']:<6} ${stat['Q1']:>9,.0f}  ${stat['Q2']:>10,.0f}   ${stat['Q3']:>9,.0f}  ${stat['IQR']:>9,.0f}  ${stat['Mean']:>9,.0f}")
print("="*85)

# Find key insights
highest_mean = max(stats_summary, key=lambda x: x['Mean'])
lowest_mean = min(stats_summary, key=lambda x: x['Mean'])
highest_median = max(stats_summary, key=lambda x: x['Q2'])
lowest_median = min(stats_summary, key=lambda x: x['Q2'])

print("\n📊 KEY INSIGHTS:")
print(f"  • Highest Average Bill:   {highest_mean['Admission Type']} (${highest_mean['Mean']:,.2f})")
print(f"  • Lowest Average Bill:    {lowest_mean['Admission Type']} (${lowest_mean['Mean']:,.2f})")
print(f"  • Highest Median Bill:    {highest_median['Admission Type']} (${highest_median['Q2']:,.2f})")
print(f"  • Lowest Median Bill:     {lowest_median['Admission Type']} (${lowest_median['Q2']:,.2f})")

# Calculate the difference between highest and lowest
mean_diff = highest_mean['Mean'] - lowest_mean['Mean']
print(f"  • Difference (Highest - Lowest Mean): ${mean_diff:,.2f}")
print("="*85)


# In[48]:


#------- Billing Amount by Gender -------

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Billing Amount', data=df_clean)
plt.title('Billing Amount by Gender')
plt.xlabel('')
plt.ylabel('Billing Amount')
plt.show()


# In[49]:


#------- Billing Amount by Test Results -------

plt.figure(figsize=(10, 6))
sns.boxplot(x='Test Results', y='Billing Amount', data=df_clean)
plt.title('Billing Amount by Test Results')
plt.xlabel('Test Results')
plt.ylabel('Billing Amount')
plt.show()


# # Correlation Analysis

# In[51]:


#------- Simple Correlation Results -------

# Select only numerical columns
numerical_df = df_clean.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numerical_df.corr()

# Display heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print correlation matrix
print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
print(correlation_matrix.round(3))

# Find strongest correlations
print("\n" + "="*60)
print("STRONGEST CORRELATIONS")
print("="*60)

# Get correlations with Billing Amount (if it exists)
if 'Billing Amount' in correlation_matrix.columns:
    billing_corr = correlation_matrix['Billing Amount'].sort_values(ascending=False)
    billing_corr = billing_corr[billing_corr.index != 'Billing Amount']
    print("\n📊 Correlations with Billing Amount:")
    for var, corr in billing_corr.items():
        print(f"  • {var}: {corr:.3f}")

# Get correlations with Age (if it exists)
if 'Age' in correlation_matrix.columns:
    age_corr = correlation_matrix['Age'].sort_values(ascending=False)
    age_corr = age_corr[age_corr.index != 'Age']
    print("\n📊 Correlations with Age:")
    for var, corr in age_corr.items():
        print(f"  • {var}: {corr:.3f}")

print("\n" + "="*60)


# # Time Series Analysis

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Ensure Date of Admission is datetime
df_clean['Date of Admission'] = pd.to_datetime(df_clean['Date of Admission'])

# ============================================
# 1. MONTHLY TREND (Connected Scatter Plot)
# ============================================

# Group by month
monthly_admissions = df_clean.groupby(pd.Grouper(key='Date of Admission', freq='M')).size()
monthly_data = monthly_admissions.reset_index()
monthly_data.columns = ['Date', 'Admissions']

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create connected scatter plot
scatter = ax.scatter(monthly_data['Date'], monthly_data['Admissions'], 
                     c=monthly_data.index, cmap='viridis', 
                     s=100, alpha=0.8, zorder=3)

# Connect points with lines
ax.plot(monthly_data['Date'], monthly_data['Admissions'], 
        color='teal', linewidth=2, alpha=0.6, zorder=2)



# Set y-axis to start from 0
ax.set_ylim(0, monthly_data['Admissions'].max() * 1.15)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45, ha='right')

# Add average line
avg = monthly_data['Admissions'].mean()
ax.axhline(y=avg, color='red', linestyle='--', linewidth=1.5, 
           label=f'Average: {avg:.0f} admissions/month')

# Add title and labels
plt.title('Monthly Admissions Trend (2019-2024) - Connected Scatter Plot', 
          fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Admissions', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Time Progression', fontsize=10)

# Manually set tick labels to show years
# Calculate positions and labels
import numpy as np
num_points = len(monthly_data)
tick_positions = [0, int(num_points*0.25), int(num_points*0.5), int(num_points*0.75), num_points-1]
tick_labels = []

for pos in tick_positions:
    if pos < num_points:
        date = monthly_data.iloc[pos]['Date']
        tick_labels.append(date.strftime('%Y'))  # Just year
        # or use '%b %Y' for "Jan 2020" format

cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()
plt.show()

# ============================================
# 2. YEARLY TREND (Connected Scatter Plot)
# ============================================

# Group by year
yearly_admissions = df_clean.groupby(df_clean['Date of Admission'].dt.year).size()
yearly_data = yearly_admissions.reset_index()
yearly_data.columns = ['Year', 'Admissions']

fig, ax = plt.subplots(figsize=(12, 6))

# Create connected scatter plot
scatter = ax.scatter(yearly_data['Year'], yearly_data['Admissions'], 
                     c=yearly_data['Year'], cmap='coolwarm', 
                     s=200, alpha=0.8, zorder=5)

# Connect points with lines
ax.plot(yearly_data['Year'], yearly_data['Admissions'], 
        color='steelblue', linewidth=2.5, alpha=0.7, zorder=2)

# Add value labels
for idx, row in yearly_data.iterrows():
    ax.annotate(f'{int(row["Admissions"]):,}', 
               (row['Year'], row['Admissions']),
               textcoords="offset points", xytext=(0, 10), 
               ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, yearly_data['Admissions'].max() * 1.15)

# Add average line
avg_yearly = yearly_data['Admissions'].mean()
ax.axhline(y=avg_yearly, color='red', linestyle='--', linewidth=1.5, 
           label=f'Average: {avg_yearly:.0f} admissions/year')

plt.title('Total Admissions by Year (2019-2024) - Connected Scatter Plot', 
          fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Admissions', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Year Progression', fontsize=10)



# Add start and end markers
ax.plot(yearly_data.iloc[0]['Year'], yearly_data.iloc[0]['Admissions'], 
        marker='o', markersize=12, color='green', label='Start')
ax.plot(yearly_data.iloc[-1]['Year'], yearly_data.iloc[-1]['Admissions'], 
        marker='o', markersize=12, color='red', label='End')

plt.tight_layout()
plt.show()

# ============================================
# 3. YEAR-OVER-YEAR CHANGE
# ============================================

years = sorted(yearly_data['Year'].values)
changes = [0]  # First year has no change
for i in range(1, len(years)):
    changes.append(yearly_data[yearly_data['Year'] == years[i]]['Admissions'].values[0] - 
                   yearly_data[yearly_data['Year'] == years[i-1]]['Admissions'].values[0])

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if c >= 0 else 'red' for c in changes]
bars = ax.bar(years, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

for bar in bars:
    height = bar.get_height()
    if height >= 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'+{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height - 15,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(min(changes) * 1.3, max(changes) * 1.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.title('Year-over-Year Change in Admissions', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Change in Admissions', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ============================================
# 4. MONTHLY PATTERN (Seasonality)
# ============================================

# Group by month number (1-12)
monthly_pattern = df_clean.groupby(df_clean['Date of Admission'].dt.month).size()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(month_names, monthly_pattern.values, color='coral', alpha=0.8, edgecolor='black', linewidth=1)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

avg_monthly = monthly_pattern.mean()
ax.axhline(y=avg_monthly, color='red', linestyle='--', linewidth=1.5, 
           label=f'Average: {avg_monthly:.0f}')

plt.title('Seasonal Pattern: Average Admissions by Month', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Number of Admissions', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ============================================
# 5. SUMMARY STATISTICS
# ============================================

print("\n" + "="*80)
print("TREND ANALYSIS - COMPLETE SUMMARY")
print("="*80)

# Basic statistics
total_admissions = monthly_data['Admissions'].sum()
total_months = len(monthly_data)
avg_monthly = total_admissions / total_months
max_month = monthly_data['Admissions'].max()
min_month = monthly_data['Admissions'].min()
busiest_month = monthly_data.loc[monthly_data['Admissions'].idxmax(), 'Date']
quietest_month = monthly_data.loc[monthly_data['Admissions'].idxmin(), 'Date']

print(f"\n📊 OVERALL STATISTICS:")
print(f"  • Total Admissions (2019-2024):  {total_admissions:,}")
print(f"  • Total Months:                  {total_months}")
print(f"  • Average per Month:             {avg_monthly:.1f}")
print(f"  • Busiest Month:                 {busiest_month.strftime('%B %Y')} ({max_month:,} admissions)")
print(f"  • Quietest Month:                {quietest_month.strftime('%B %Y')} ({min_month:,} admissions)")

# Yearly statistics
print("\n📊 YEARLY STATISTICS:")
print("-"*50)
for year in sorted(yearly_data['Year'].values):
    count = yearly_data[yearly_data['Year'] == year]['Admissions'].values[0]
    pct = (count / total_admissions) * 100
    print(f"  {year}: {count:>5,} admissions ({pct:>5.1f}% of total)")

# Year-over-year change
print("\n📈 YEAR-OVER-YEAR CHANGE:")
print("-"*50)
for i in range(1, len(years)):
    prev = yearly_data[yearly_data['Year'] == years[i-1]]['Admissions'].values[0]
    curr = yearly_data[yearly_data['Year'] == years[i]]['Admissions'].values[0]
    change = curr - prev
    pct = (change / prev) * 100
    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    print(f"  {years[i-1]} → {years[i]}: {direction} {change:+,} admissions ({pct:+.1f}%)")

# Seasonal pattern
print("\n📅 SEASONAL PATTERN (Busiest vs Quietest Months):")
print("-"*50)
busiest_month_num = monthly_pattern.idxmax()
quietest_month_num = monthly_pattern.idxmin()
print(f"  • Busiest Month:   {month_names[busiest_month_num-1]} ({monthly_pattern.max():.0f} avg admissions)")
print(f"  • Quietest Month:  {month_names[quietest_month_num-1]} ({monthly_pattern.min():.0f} avg admissions)")
print(f"  • Difference:      {monthly_pattern.max() - monthly_pattern.min():.0f} admissions")

# Trend direction
first_half = monthly_data.iloc[:len(monthly_data)//2]['Admissions'].mean()
second_half = monthly_data.iloc[len(monthly_data)//2:]['Admissions'].mean()
trend = "INCREASING" if second_half > first_half else "DECREASING"
trend_pct = ((second_half - first_half) / first_half) * 100

print(f"\n📈 TREND DIRECTION:")
print("-"*50)
print(f"  • First Half Average:  {first_half:.1f} admissions/month")
print(f"  • Second Half Average: {second_half:.1f} admissions/month")
print(f"  • Trend:               {trend} ({trend_pct:+.1f}%)")

if abs(trend_pct) > 10:
    print(f"  ⚠️ Significant {trend} trend detected!")

print("\n" + "="*80)


# # Length of Stay Analysis

# In[55]:


#------- Calculate Length of Stay -------

df_clean['Length of Stay'] = (df_clean['Discharge Date'] - df_clean['Date of Admission']).dt.days


# In[56]:


#------- Distribution of Length of Stay -------
plt.figure(figsize=(10, 6))
ax = sns.histplot(df_clean['Length of Stay'], bins=30, kde=True, color='coral', alpha=0.7)

# Add vertical lines for mean and median
mean_los = np.mean(df_clean['Length of Stay'])
median_los = np.median(df_clean['Length of Stay'])
plt.axvline(mean_los, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_los:.1f} days')
plt.axvline(median_los, color='green', linestyle='--', linewidth=2, label=f'Median: {median_los:.1f} days')

plt.title('Distribution of Length of Stay (Days)', fontsize=14, fontweight='bold')
plt.xlabel('Length of Stay (Days)', fontsize=12)
plt.ylabel('Patients (Frequency)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[57]:


#------- Length of Stay by Medical Condition -------

plt.figure(figsize=(14, 6))
ax = sns.boxplot(x='Medical Condition', y='Length of Stay', data=df_clean, palette='Set2')
plt.title('Length of Stay by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Medical Condition', fontsize=12)
plt.ylabel('Length of Stay (Days)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[58]:


#------- Length of Stay by Admission Type -------

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Admission Type', y='Length of Stay', data=df_clean, palette='viridis')
plt.title('Length of Stay by Admission Type', fontsize=14, fontweight='bold')
plt.xlabel('Admission Type', fontsize=12)
plt.ylabel('Length of Stay (Days)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# # Cross-tabulation Analysis

# In[60]:


#------- Medical Condition vs Test Results -------

ct = pd.crosstab(df_clean['Medical Condition'], df_clean['Test Results'])
print("\n--- Medical Condition vs Test Results ---")
print(ct)

plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Count'})
plt.title('Number of Medical Condition by Test Results', fontsize=14, fontweight='bold')
plt.xlabel('Test Results', fontsize=12)
plt.ylabel('Medical Condition', fontsize=12)
plt.tight_layout()
plt.show()


# In[61]:


#------- Gender vs Medical Condition -------

ct_gender_condition = pd.crosstab(df_clean['Gender'], df_clean['Medical Condition'])
print("\n--- Gender vs Medical Condition ---")
print(ct_gender_condition)

plt.figure(figsize=(12, 6))
ct_gender_condition.plot(kind='bar', stacked=True, colormap='Set3')
plt.title('Gender Distribution by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Patients', fontsize=12)
plt.legend(title='Medical Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# # Summary Statistics for All Columns

# In[63]:


#------- Summary Statistics -------

print("\n--- Summary Statistics for Numerical Columns ---")
print(df_clean.describe())

print("\n--- Additional Statistics (Q1, Q3, IQR, Skewness, Kurtosis) ---")
for col in numerical_cols:
    data = df_clean[col].values
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = sum((xi - mean_val) ** 3 for xi in data) / (len(data) * std_val ** 3) if std_val > 0 else 0
    kurtosis = sum((xi - mean_val) ** 4 for xi in data) / (len(data) * std_val ** 4) - 3 if std_val > 0 else 0
    print(f"\n{col}:")
    print(f"  Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

print("\n--- Summary Statistics for Categorical Columns ---")
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df_clean[col].value_counts())

print("\n--- Analysis Complete ---")

