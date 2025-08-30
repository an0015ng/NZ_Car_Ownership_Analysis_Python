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

<img width="780" height="309" alt="image" src="https://github.com/user-attachments/assets/2b69e4aa-40d6-448c-b9cc-a96dbbad145f" />

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

I start by examining the basic structure and identifying any data quality issues that need attention.
