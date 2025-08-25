# New Zealand Car Purchase Data Analysis | Python & Pandas

## Executive Summary

This project conducts a comprehensive analysis of **New Zealand car ownership data encompassing 21 subjects** to demonstrate foundational data analysis skills through systematic data understanding, cleaning, and exploratory analysis using Python and pandas. Through advanced data manipulation techniques including **missing value analysis**, **feature engineering with datetime operations**, and **statistical outlier detection using IQR methods**, I transform raw car purchase records into meaningful insights about demographics, vehicle preferences, and market trends.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Phase 1: Data Cleaning & Quality Assessment](#phase-1-data-cleaning--quality-assessment)
- [Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
- [Phase 3: Data Visualization & Insights](#phase-3-data-visualization--insights)
- [Key Findings & Business Insights](#key-findings--business-insights)
- [Skills Demonstrated](#skills-demonstrated)
- [Conclusion](#conclusion)

---

## Project Overview

### Business Context
Understanding car ownership patterns and market trends is crucial for automotive businesses, insurance companies, and policy makers. This analysis demonstrates how raw vehicle registration data can be transformed into actionable insights about consumer preferences, demographic patterns, and market segmentation opportunities.

### Project Goal
I aim to showcase essential data analysis skills by conducting a comprehensive examination of New Zealand car purchase data, demonstrating proficiency in:
- **Data Quality Assessment**: Systematic evaluation of data completeness and reliability
- **Data Cleaning & Preparation**: Professional-grade data preprocessing techniques
- **Feature Engineering**: Creation of meaningful derived variables for enhanced analysis
- **Exploratory Data Analysis**: Statistical investigation of patterns and relationships
- **Data Visualization**: Professional chart creation for insights communication

---

## Dataset Overview

**Source**: [Kaggle - Data for NZ Vehicle Info Analyze](https://www.kaggle.com/datasets/mlinnz/data-for-nz-vehicle-info-analyze/data)

![Car Dataset Preview](image.jpg)

**Key Statistics**: 21 records × 16 columns | Age range: 28-68 years | Purchase years: 1992-2022 | Multiple countries of origin represented

The dataset represents real-world car ownership records with comprehensive demographic and vehicle information:

**Data Structure**: Demographics → Purchase History → Vehicle Specifications → Geographic Origin  
**Coverage Areas**: Gender distribution, birth years, registration timeline, purchase patterns  
**Vehicle Metrics**: Car make/model, fuel economy, seating capacity, vehicle classifications  
**Geographic Scope**: International car origins including Japan, UK, Germany, Austria, Sweden, Australia, United States

---

# Phase 1: Data Cleaning & Quality Assessment

I begin with comprehensive data quality assessment to ensure reliable analysis, establishing the foundation for all subsequent analytical work.

## 1.1. Missing Values Analysis

I start by examining the basic structure and identifying any data quality issues that need attention.
