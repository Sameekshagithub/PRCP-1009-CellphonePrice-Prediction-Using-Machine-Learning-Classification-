PRCP-1009 : Cellphone Price Range Prediction Using Machine Learning

PROJECT TITLE: PRCP-1009: Cellphone Price Range Prediction Using Machine Learning
TEAM CODE: PTID-AIE-JUL-26-11142
PROJECT CODE: PRCP-1009
PROJECT TYPE: Data Science Capstone Project
DATASET SOURCE: DataMites Capstone Project Dataset
DATASET NAME: Mobile Price Classification (datasets_11167_15520_train.csv)
DATASET LINK: https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1009-CellphonePrice.zip
SUBMITTED BY: Sameeksha


PROBLEM STATEMENT

The objective of this project is to analyze mobile phone specifications and
develop a machine learning classification model to predict the price range of
a mobile phone based on its features. The project also involves comparing
multiple classification models, identifying the most influential features
affecting price, and providing business insights for effective pricing
decisions.


PROJECT OBJECTIVES

1. Perform Exploratory Data Analysis (EDA) to understand the dataset.
2. Preprocess and prepare the data for model development.
3. Build and train multiple classification models.
4. Evaluate and compare model performance using appropriate metrics.
5. Select the best-performing model for production.
6. Analyze feature importance to identify the key factors influencing mobile
   phone price.
7. Provide business insights and recommendations based on the model results.
8. Summarize the challenges faced and the techniques used during the project.


DATASET

The dataset (datasets_11167_15520_train.csv) contains 2,000 mobile phone
records with 20 specification features and one target variable, price_range:

0 - Low Cost
1 - Medium Cost
2 - High Cost
3 - Very High Cost

Features include battery_power, blue, clock_speed, dual_sim, fc, four_g,
int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h,
sc_w, talk_time, three_g, touch_screen, and wifi.


NOTEBOOK STRUCTURE

1. Problem Statement & Project Objective
2. Import Python Libraries
3. Upload the Dataset And Domain Analysis
4. Basic Checks (shape, dtypes, missing values, duplicates, class counts)
5. Exploratory Data Analysis (EDA) - Data Analysis Report (Task 1)
6. Data Preparation for Modelling (Task 2) - train/test split, feature scaling
7. Model Building & Training
   7.1 Logistic Regression
   7.2 Decision Tree Classifier
   7.3 Random Forest Classifier
   7.4 K-Nearest Neighbours (KNN)
   7.5 Support Vector Machine (SVM)
   7.6 Gradient Boosting Classifier
8. Hyperparameter Tuning (GridSearchCV on the best-performing model)
9. Model Comparison Report
10. Feature Importance
11. Model Testing on New Mobile Specifications
12. Business Insights (Task 3)
13. Challenges Faced
14. Final Conclusion
15. Project Links


RESULTS SUMMARY

Six classification models were trained and evaluated using Accuracy,
Precision, Recall, F1-Score, Classification Report, and Confusion Matrix.
Logistic Regression achieved the highest accuracy and F1-score among all
models and was further optimized using GridSearchCV. It is recommended as
the final production model for predicting mobile phone price range.

Feature Importance analysis (based on Random Forest) identified RAM, battery
power, display resolution (px_height and px_width), and internal memory as
the specifications with the greatest influence on price range. Connectivity
features such as Bluetooth, WiFi, and dual SIM had comparatively little
impact, since they are present on nearly all modern phones.


HOW TO RUN

1. Install the required libraries:
   pip install numpy pandas matplotlib seaborn scikit-learn

2. Place datasets_11167_15520_train.csv in the same folder as the notebook.

3. Open PRCP_1009_Cellphone_Price_Range_Prediction_Using_Machine_Learning_.ipynb
   in Jupyter Notebook, JupyterLab, or VS Code.

4. Run all cells in order, from Section 1 through Section 15.


PROJECT LINKS

GitHub Repository:
https://github.com/Sameekshagithub/PRCP-1009-CellphonePrice-Prediction-Using-Machine-Learning-Classification-.git

Google Drive (Presentation):
https://docs.google.com/presentation/d/1RE
