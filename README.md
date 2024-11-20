# Donor Churn Prediction Project

## Overview
The Donor Churn Prediction project aims to help organizations retain donors by leveraging data analytics and predictive modeling. By identifying at-risk donors early, organizations can implement targeted engagement strategies to enhance donor retention, maximize revenue, and improve campaign efficiency. My goal is to get a hands-on exprience on a Data Anlaystics Project with donors and fundraise data for specific cause such in research, teaching and patient care.

The Donor Churn Prediction project utilizes advanced data analytics, predictive modeling, and ETL pipelines to identify at-risk donors, enabling targeted engagement strategies. Leveraging tools like Tableau for interactive dashboards, AWS SageMaker and S3 for scalable ML deployment, and privacy-compliant workflows, this project enhances retention, maximizes revenue, and provides hands-on experience in data-driven decision-making for fundraising in research, teaching, and patient care.


## Business Context
**Donor Retention**: Retaining existing donors is critical for sustainability. Insights from this project support donor relationship management.

**Scalable Strategies**: The predictive models developed are scalable and can be used across campaigns to provide broad and actionable insights.

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Data Sources](#data-acquisition)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Development](#model-development)
8. [Model Evaluation](#model-evaluation)
9. [Deployment](#deployment)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction
Donor churn significantly impacts fundraising efforts. Predicting churn enables organizations to implement targeted retention strategies.

## Objective
**Improve Donor Retention** : Retaining donors is more cost-effective than acquiring new ones. This project focuses on identifying at-rsik donors to target/morivate/re-engage them.
  
### Data Sources
- **Primary Files:**
  - **Donor Data:** Contains comprehensive donor information, including demographics, giving history, and wealth ratings. This serves as the core dataset for donor churn analysis.
  - **Contact Reports:** Includes interaction details with donors, such as methods of contact and outcomes, which will be used to derive engagement metrics.
- **Source:**
  - Kaggle: [Fundraising Data](https://www.kaggle.com/datasets/michaelpawlus/fundraising-data?select=data_science_for_fundraising_donor_data.csv)

## Data Exploration and Preprocessing
 
Donor Data Insights:

Key columns include ZIPCODE, AGE, MARITAL_STATUS, PrevFYGiving, CurrFYGiving, TotalGiving, and others.
Missing values were handled using median imputation and default categories.

Contact Reports Insights:

Key columns include Staff Name, Method, Outcome, and Summary.
This dataset provided insights into communication methods and outcomes.

Challenges: 
- Missing data in critical columns like donor age, marital status, and wealth rating was addressed by:
Imputing missing numeric values with median or mode.
- Filling categorical columns with "Unknown."
- Dropping columns with excessive missing data that lacked meaningful information (e.g., wealth rating).
- Monetary Columns: Cleaned monetary columns by removing dollar signs and converting them into numeric format.

## Feature Engineering

In this project, we meticulously crafted features to enhance the predictive power of our models. The key features developed include:

**ScaledFrequency**: 
Normalized frequency of donor contributions, highlighting consistency in giving behavior. This feature contributed the most to churn prediction (61.3% importance).

**LogMonetary**:
Log-transformed monetary value of contributions to stabilize variance and enhance interpretability, with a significant feature importance of 35.7%.

**RecencyCategory**:
Categorized recency of donations to capture the impact of time since the last contribution, contributing 2.9% to the model's predictions.

These features were carefully engineered to provide the Gradient Boosting model with interpretable and meaningful input for predicting churn.

## Exploratory Data Analysis (EDA)

EDA was conducted to uncover patterns and relationships within the data, focusing on understanding donor behavior and characteristics that correlate with churn. Key findings include:

Distribution Analysis: Donations showed a right-skewed distribution, prompting log transformation for normalization.
Churn Behavior: High recency and low frequency were strong indicators of churn.

Correlation Analysis: Monetary value was moderately correlated with donor retention, underscoring its importance in predicting churn.

Visualizations, including histograms, scatterplots, and heatmaps, were used to explore data distributions and relationships, guiding feature engineering and model development.


## Model Development

Multiple machine learning models were trained and evaluated to predict donor churn effectively. The models include:

- **Random Forest**: Achieved 93% accuracy but had slightly lower AUC (0.93) compared to Gradient Boosting and XGBoost.
- **Gradient Boosting**: Selected as the final model due to its balanced performance metrics and superior AUC (0.97).
- **Logistic Regression**: A baseline model with 92% accuracy and perfect recall, useful for benchmarking despite potential overfitting.
- **XGBoost**: Comparable to Gradient Boosting in performance (AUC = 0.97) but more computationally intensive.
Hyperparameter tuning was performed using grid search to optimize model performance, focusing on learning rate, depth, and estimators.

 ### Feature Importance: Identified top predictors for churn:
Monetary contributions (LogMonetary) were the strongest indicator.
Frequency of contributions and composite interaction features also played key roles.

 ### Clustering Analysis
Applied KMeans clustering to segment donors into behavior-based groups:
- Cluster 1: Low-frequency, low-value donors.
- Cluster 2: High-value, highly engaged donors.
- Cluster 3: Sporadic donors with moderate contributions.
Insights from clusters provided guidance for tailoring engagement strategies.

## Model Evaluation

Evaluation metrics provided insights into model effectiveness and allowed us to compare their performance. Key metrics include:

Accuracy: The overall classification accuracy for Gradient Boosting was 93%.
Precision and Recall: Precision (0.89) and Recall (0.98) demonstrated the model's ability to minimize false negatives and positives for churned donors.
F1-Score: Balanced performance for churned donors, with an F1-score of 0.93.
ROC-AUC: Gradient Boosting and XGBoost outperformed others with an AUC of 0.97, indicating excellent discriminatory power.
Confusion matrix analysis highlighted the model's ability to correctly classify churned donors (True Positives: 5946) while maintaining low false negatives (96).

## Deployment

The final Gradient Boosting model was prepared for deployment:

Model Saving: The model was serialized as gradient_boosting_model.pkl for future use without retraining.
Prediction Simulation: A simulated donor example demonstrated the model's ability to accurately predict churn probabilities.
High-Risk Donors: Donors with a churn probability > 0.8 were identified and saved in a file (high_risk_donors.csv) for proactive engagement campaigns.
The deployment workflow ensures the model's integration with CRM systems for real-time predictions and strategic decision-making.

## Conclusion

**The Gradient Boosting model emerged as the best solution for predicting donor churn**, offering:

High accuracy (93%) and robust AUC (0.97), ensuring reliable predictions.
Interpretability through feature importance analysis, highlighting actionable insights like the impact of ScaledFrequency and LogMonetary.
Practical utility by identifying high-risk donors for targeted retention strategies.
This project demonstrates how machine learning can be leveraged to improve donor retention and optimize engagement strategies.

Insights from Confusion Matrix:

True Positives: 5,946 donors were correctly identified as likely to churn, ensuring targeted intervention.
False Negatives: Only 96 churned donors were misclassified as non-churned, minimizing missed opportunities.
Precision and Recall Balance: Achieving high recall (98%) while maintaining strong precision (89%) ensures that the model effectively identifies most at-risk donors without overwhelming the system with false alarms.

**High-Risk Donors Identified**

A list of 5,201 high-risk donors was created, containing donors with a predicted churn probability greater than 80%.
Key examples of high-risk donors include:
Donor IDs: 9, 29, 34, with churn probabilities exceeding 90%.
The high-risk donors have been saved in a file titled high_risk_donors.csv, enabling targeted outreach campaigns to retain these donors.

**Actionable Insights**:

- Engagement Campaigns: Focus retention efforts on the 5,201 high-risk donors identified by the model. These donors are prime candidates for personalized outreach programs.
  
- Monetary Contributions: Introduce strategies to boost donor contributions for those with low monetary value, such as tailored campaigns emphasizing impact.
  
- Frequency Interventions: Implement targeted initiatives to increase contribution frequency among moderate- to high-risk donors.
  
- Proactive CRM Integration: The high_risk_donors.csv file is ready for integration with donor management systems to streamline outreach efforts.

## References

- Data Sources
Donor Data: The primary dataset used for analysis, containing donor information and historical contribution data.
Contact Reports: Additional data providing context for donor interactions and engagement.

- Tools and Technologies
Python: Core programming language for data analysis and model development.
Libraries:
Scikit-learn: Used for machine learning model implementation, evaluation, and feature importance analysis.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical computations.


- External Libraries
XGBoost: For implementing the XGBoost model.
Gradient Boosting: For building and tuning the final model.
