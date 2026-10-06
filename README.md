# Healthcare Dataset – Exploratory Data Analysis

## Project Overview

This project presents an exploratory data analysis (EDA) of a healthcare dataset. The analysis applies descriptive statistics and data visualisation techniques to explore the characteristics, distributions, variability, and relationships within the dataset.

## Dataset

The dataset contains **55,500 patient records and 15 variables** covering areas such as:

- Patient demographics
- Medical conditions
- Hospital admissions
- Insurance providers
- Medication
- Billing amounts
- Test results
- Admission and discharge dates

The dataset was obtained from Kaggle.

## Analysis Performed

The analysis includes:

- Data types and dataset structure
- Missing-value inspection
- Duplicate-record inspection
- Categorical-value validation
- Numerical data validation
- Measures of location, including mean and median
- Measures of variability, including variance and standard deviation
- Distribution analysis
- Categorical frequency analysis
- Covariance analysis
- Pearson correlation analysis
- Data visualisation using charts and statistical plots

## Key Variables

The main numerical variables analysed are:

- **Age**
- **Billing Amount**

Categorical variables analysed include:

- Gender
- Blood Type
- Medical Condition
- Insurance Provider
- Admission Type
- Medication
- Test Results

## Visualisations

The project includes visualisations such as:

- Line plots
- Bar charts
- Stacked bar charts
- Histograms
- Kernel Density Estimate (KDE) plots
- Boxplots
- Covariance heatmaps
- Pearson correlation heatmaps
- Regression plots

The visualisations are stored in the `figures` folder.

## Technologies Used

- Python
- JupyterLab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

## Project Structure

Healthcare_Analysis/
│
├── Healthcare_Analysis.ipynb
├── README.md
├── figures/
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── ...
└── data/
    └── healthcare_dataset.csv

## How to Run

1. Download or clone this repository.
2. Open the project in JupyterLab.
3. Open `Healthcare_Analysis.ipynb`.
4. Run the notebook cells from beginning to end.