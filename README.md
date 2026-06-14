🚀 E-Commerce 360° Analytics & Customer Churn Prediction

An end-to-end Data Science project built using Amazon India Sales Data that combines Data Cleaning, Exploratory Data Analysis (EDA), RFM Customer Segmentation, Machine Learning-based Churn Prediction, Explainable AI (SHAP), and an Interactive Streamlit Dashboard.

This project helps businesses identify valuable customers, understand customer behavior, and predict which customers are likely to churn so retention strategies can be implemented proactively.


📌 Project Overview

Customer retention is one of the most important challenges in e-commerce. Acquiring a new customer is significantly more expensive than retaining an existing one.

This project provides a complete analytics solution by:

•Cleaning and transforming raw sales data
•Performing Exploratory Data Analysis (EDA)
•Creating RFM customer segments
•Engineering churn-related features
•Building and comparing multiple ML models
•Explaining predictions using SHAP
•Deploying insights through a Streamlit dashboard

🎯 Business Problem

Can we identify customers who are likely to stop purchasing before they actually leave?

By predicting churn early, businesses can:

•Increase customer retention
•Reduce revenue loss
•Improve marketing efficiency
•Target high-value at-risk customers


🛠 Tech Stack

Programming
•Python

Data Analysis
•Pandas
•NumPy

Data Visualization
•Matplotlib
•Seaborn

Machine Learning
•Scikit-Learn
•XGBoost
•SMOTE (Imbalanced Learning)

Explainable AI
•SHAP

Deployment
•Streamlit

🔍 Data Cleaning & Preprocessing
Performed:

✅ Standardized column names

✅ Removed unnecessary columns

✅ Converted date columns

✅ Handled missing values

✅ Created delivery and cancellation flags

✅ Revenue calculation

✅ Created time-based features

•Month
•Week
•Day of Week

👥 Customer Analytics using RFM

RFM stands for:

Recency	    -    Days since last purchase

Frequency	 -     Number of purchases

Monetary	   -   Total customer spending


🔶Customer Segments

Customers were categorized into:

🏆 Champions
💎 Loyal Customers
⚠️ At Risk
🔍 Needs Attention
❌ Lost Customers

These segments help businesses create personalized retention campaigns.

🔶Models Trained

1. Logistic Regression
Fast baseline model
Highly interpretable

2. Random Forest
Ensemble learning
Handles nonlinear patterns

3. XGBoost
Gradient boosting algorithm
State-of-the-art performance


📊 Model Evaluation Metrics

The models were evaluated using:

•Accuracy
•Precision
•Recall
•F1 Score
•ROC-AUC

The best-performing model was automatically selected based on ROC-AUC score.

🧠 Explainable AI with SHAP

SHAP was used to explain:

•Why a customer was predicted to churn

•Which features increased churn risk

•Which features reduced churn risk

This makes the model transparent and business-friendly.
