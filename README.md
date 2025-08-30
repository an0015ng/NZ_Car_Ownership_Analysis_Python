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

# Phase 2: Exploratory Data Analysis

Now that I have clean, reliable data, it's time to dig deeper and understand what the data is actually telling us. This phase is about exploring patterns, relationships, and getting a feel for the characteristics of New Zealand's car market over the past 50+ years.

## 2.1. Demographic Analysis & Market Segmentation

**Understanding the Basics**: Before diving into complex market trends, I need to understand the fundamental characteristics of the dataset - who's buying cars, what types of cars they're buying, and how the market is distributed.

I set up a systematic approach to analyze all the categorical variables in the dataset. Since some variables have just a few categories (like Gender) while others have hundreds (like Car Model), I needed different visualization strategies:

```python
sns.set_style("whitegrid")

#Analyze categorical variables
categorical_columns = ['Gender', 'Car_status', 'Car_Make', 'Car_Model',
'Vehicle_type', 'Vehicle_shape', 'Country_of_origin']

for col in categorical_columns:
if col in df_clean.columns:
try:
# Print the header
print(f"\n{col.upper()} - Value Counts:")

# Perform calculations
value_counts = df_clean[col].value_counts()
percentages = df_clean[col].value_counts(normalize=True) * 100
unique_count = df_clean[col].nunique()

# Modify summary based on number of unique values
if unique_count <= 5:
    # Show all categories for small number of unique values
    summary = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages.round(2)
    })
    print(summary.head(10))
else:
    # Show top 9 + Others for large number of unique values
    # [Additional code for handling many categories...]
```

**Smart Visualization Strategy**: The code automatically chooses the best chart type:
- **Pie charts** for variables with 5 or fewer categories (like Gender, Car Status)
- **Horizontal bar charts** with "Top 9 + Others" for variables with many categories (like Car Make, Car Model)

This prevents charts from becoming cluttered and unreadable while still showing the most important patterns.

### Key Demographic Findings

**Gender Distribution**:

<p align="center">
  <img width="330" height="358" alt="image" src="https://github.com/user-attachments/assets/ecbefb42-c7c1-45ab-a3c4-6207eff9b1d1" />
</p>

The gender split is remarkably balanced - 50.9% male and 49.1% female car owners. This near-perfect 50/50 split suggests that car ownership in New Zealand isn't significantly skewed by gender, which is interesting for market segmentation purposes. Both men and women are equally represented in the car-buying market.

**Car Status Analysis**:

<p align="center">
  <img width="274" height="316" alt="image" src="https://github.com/user-attachments/assets/dd8fcbf4-aaf3-48bb-ac93-9f98a6f87053" />
</p>

This 70/30 split between old and new cars has major implications for automotive businesses - it shows the used car market is much larger than the new car market.

### Market Landscape & Brand Competition

**Car Make Analysis**: 

<p align="center">
<img width="379" height="383" alt="image" src="https://github.com/user-attachments/assets/0b6f4ce0-7c66-419d-809d-87cbe5cf1882" />
</p>
This reveals a fascinating competitive landscape in New Zealand's automotive market. The top three brands are remarkably close: Ford leads with 9.87% market share, followed closely by Toyota (9.55%) and Mitsubishi (9.05%). This tight competition among the leaders suggests a healthy, competitive market without extreme dominance by any single brand.

**What's Really Interesting**: The "Others" category accounts for 44.13% of the market, which means the car market is incredibly diverse - there are 245 different car makes in the dataset! This suggests New Zealanders have access to a wide variety of international brands, not just the major global players.















