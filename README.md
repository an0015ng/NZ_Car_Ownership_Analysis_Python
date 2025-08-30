# New Zealand Car Purchase Data Analysis | Python & Pandas

## Executive Summary

This project analyzes **300,000 New Zealand car ownership records from 1970-2023** to showcase my data analysis skills using Python and pandas. I worked through the complete data analysis process, starting from cleaning messy data to uncovering interesting patterns about car ownership, demographics, and how different countries have dominated the car market over five decades. This analysis demonstrates my ability to handle large datasets and extract meaningful business insights that car dealerships, manufacturers, and automotive businesses can actually use.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Phase 1: Data Cleaning & Quality Assessment](#phase-1-data-cleaning--quality-assessment)
- [Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
- [Phase 3: Market Dominance Analysis](#phase-3-market-dominance-analysis)
- [Skills Demonstrated](#skills-demonstrated)
- [Project Files & Access](#project-files--access)
- [Conclusion](#conclusion)

---

## Project Overview

This project helps understanding how car preferences and market trends have evolved over time. Using Python's powerful data analysis tools, I took raw automotive purchase data and turned it into clear insights about what cars people buy, when they buy them, and how global automotive competition has shifted over the decades. This work demonstrates my ability to handle large datasets systematically and extract actionable insights that businesses can use to make better decisions.

---

## Dataset Overview

**Source**: [Kaggle - Data for NZ Vehicle Info Analyze](https://www.kaggle.com/datasets/mlinnz/data-for-nz-vehicle-info-analyze/data)

Here's what the raw dataset `Car_Data.csv` looks like when I first loaded it:

<p align="center">
  <img width="1285" height="376" alt="image" src="https://github.com/user-attachments/assets/ee88473e-c1a4-4f52-8f23-c652c0638b15" />
</p>

**Key Statistics**: 300,000 records × 16 columns | Age range: 18-71 years | Purchase timeline: 1970-2023 (53 years) | International vehicle origins from 8+ countries

This dataset contains a fully synthetic yet privacy-preserving dataset that mimics New Zealand passenger-vehicle registrations and purchasesreal automotive purchase records spanning over five decades:

**Data Hierarchy**: Demographics → Purchase History → Vehicle Specifications → Country of Origin Analysis  
**Demographic Coverage**: Gender distribution, birth years (1954-2007), registration and purchase timelines  
**Vehicle Metrics**: Car make/model, fuel economy, seating capacity, vehicle types and shapes  
**Geographic Scope**: International automotive origins including Japan, Germany, UK, South Korea, US, Sweden, Australia, and others

---

# Phase 1: Data Cleaning & Quality Assessment

I begin with comprehensive data quality assessment to ensure reliable analysis, establishing the foundation for all subsequent analytical work.

## 1.1. Missing Values Analysis

**Why This Matters**: Missing data can completely skew analysis results, so I always start by understanding what's missing and why.

I started by checking if there were any missing values across all columns in the dataset. This is essential because missing data can indicate data collection issues or require special handling strategies.

```python
#Check for missing values & percentages
missing_count = df.isnull().sum()
print("\nMissing Values Percentage:")
missing_percentage = (missing_count / len(df)) * 100
missing_summary = pd.DataFrame({
'Missing_Count': missing_count,
'Missing_Percentage': missing_percentage.round(2)
}).sort_values('Missing_Percentage', ascending=False)

print(missing_summary)
```
Below is the output:

<p align="center">
  <img width="612" height="400" alt="image" src="https://github.com/user-attachments/assets/1290e877-fde1-4942-8631-9fee98430764" />
</p>

The results showed 0.0% missing values across all 16 columns. With 300,000 records and no missing values, I could proceed with confidence that the analysis wouldn't be compromised by data gaps.

## 1.2. Data Type Standardization & Optimization  

**The Problem**: When data is first loaded, pandas often assigns generic data types that aren't optimal for analysis. Dates might be stored as text, and categorical variables might be treated as generic objects.

I needed to convert the data types to make the analysis more efficient and accurate:

```python
#Convert date columns to proper datetime format
if 'Purchase_date' in df_clean.columns:
df_clean['Purchase_date'] = pd.to_datetime(df_clean['Purchase_date'], errors='coerce')

#Ensure categorical variables are properly typed
categorical_columns = ['Gender', 'Car_status', 'Car_Make', 'Car_Model',
'Vehicle_type', 'Vehicle_shape', 'Country_of_origin']

for col in categorical_columns:
if col in df_clean.columns:
df_clean[col] = df_clean[col].astype('category')
```

Converting dates to datetime format allows for proper time-series analysis later. Converting text fields to categorical data types reduces memory usage and enables more efficient grouping operations - important when working with 300,000 records.

## 1.3. Feature Engineering & Temporal Variables

**Creating More Useful Data**: Raw data often doesn't have all the information I need for analysis. I created new calculated fields that would be essential for understanding car ownership patterns.

The original dataset had birth years and purchase years, but what I really needed was current age and how long people have owned their cars:

```python
#Calculate age from birth year (current year is 2025)
current_year = 2025
if 'Year_of_birth' in df_clean.columns:
df_clean['Age'] = current_year - df_clean['Year_of_birth']

#Calculate years since purchase
if 'Purchase_year' in df_clean.columns:
df_clean['Years_since_purchase'] = current_year - df_clean['Purchase_year']

#Remove duplicates
print(f"Duplicates found: {df_clean.duplicated().sum()}")
df_clean.drop_duplicates(inplace=True)
print(f"Dataset shape after removing duplicates: {df_clean.shape}")
```

Below is the output:

<p align="center">
  <img width="612" height="62" alt="image" src="https://github.com/user-attachments/assets/9a97c997-349f-4927-988f-5583cd54dac4" />
</p>

**New Features Created**:
- **Age**: Current age of each car owner (2025 - birth year)
- **Years_since_purchase**: How long ago each car was purchased (2025 - purchase year)

**Duplicate Check**: The output showed 0 duplicates, confirming the data quality was excellent. The final clean dataset maintained all 300,000 records with 18 columns (the original 16 plus my 2 new calculated fields).

**Why These Features Matter**: Age is crucial for demographic analysis, while years_since_purchase helps understand car ownership patterns and replacement cycles. These calculated fields will be essential for the market analysis in Phase 3.

At this point, I had a clean, well-structured dataset with proper data types and useful calculated fields, ready for exploratory analysis.






