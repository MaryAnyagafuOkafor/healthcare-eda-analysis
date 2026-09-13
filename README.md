# Exploratory Data Analysis: Healthcare Dataset

## Overview
This repository contains a comprehensive exploratory data analysis (EDA) 
of a healthcare dataset containing 54,966 patient records with 15 variables.

## Dataset
- **File:** healthcare_dataset_cleaned.csv
- **Records:** 54,966 (after removing 34 rows with missing values)
- **Variables:** 15

## Key Findings

| Statistic | Billing Amount |  Age  | Room Number |
|-----------|--------------- | ----- |-------------|
| Mean      | $25,546.24     | 51.54 | 301.12 |
| Median    | $25,542.75     | 52.00 | 302.00 |
| Mode      | $14,238.32     | 38.00 | 393.00 |
| Std Dev   | $14,204.80     | 19.61 | 115.22 |
| Min       | $9.24          | 13    | 101    |
| Max       | $52,764.28     | 89    | 500    |

## Repository Contents

- `README.md` — This file
- `Exploratory Data Analysis.py` — Complete Python analysis code
- `healthcare_dataset_cleaned.csv` — Cleaned dataset
- `figures/` — All 20 visualizations

## Available Figures

| Figure | Title |
|--------|-------|
| Figure 1 | Histogram of Billing Amount |
| Figure 2 | Boxplot of Billing Amount |
| Figure 3 | Violin Plot of Billing Amount |
| Figure 4 | Histogram of Age Distribution |
| Figure 5 | Boxplot of Age by Gender |
| Figure 6 | Count Plot of Medical Conditions |
| Figure 7 | Count Plot of Admission Types |
| Figure 8 | Count Plot of Test Results |
| Figure 9 | Billing Amount by Medical Condition |
| Figure 10 | Billing Amount by Admission Type |
| Figure 11 | Correlation Matrix Heatmap |
| Figure 12 | Monthly Admissions Trend |
| Figure 13 | Yearly Admissions Trend |
| Figure 14 | Year-over-Year Change |
| Figure 15 | Seasonal Pattern |
| Figure 16 | Distribution of Length of Stay |
| Figure 17 | Length of Stay by Medical Condition |
| Figure 18 | Length of Stay by Admission Type |
| Figure 19 | Medical Condition vs Test Results |
| Figure 20 | Gender vs Medical Condition |

## How to Reproduce

1. Download the files from this repository
2. Install required packages: `pip install pandas numpy matplotlib seaborn scipy`
3. Run: `python "Exploratory Data Analysis.py"`
