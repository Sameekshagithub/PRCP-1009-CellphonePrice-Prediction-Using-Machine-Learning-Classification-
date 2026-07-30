<div align="center">

# 📱 PRCP-1009: Cellphone Price Range Prediction Using Machine Learning

### Predicting Mobile Phone Price Categories using Supervised Machine Learning Classification

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange.svg">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg">
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue.svg">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-red.svg">
<img src="https://img.shields.io/badge/Seaborn-EDA-purple.svg">
<img src="https://img.shields.io/badge/License-MIT-success.svg">

---

### 🚀 DataMites AI Engineer Capstone Project

**Project Code:** PRCP-1009  
**Team Code:** PTID-AIE-JUL-26-11142  
**Project Type:** Machine Learning Classification  
**Submitted By:** Sameeksha Rai

</div>

---

# 📖 Table of Contents

- Project Overview
- Business Problem
- Project Objectives
- Dataset Information
- Technology Stack
- Machine Learning Workflow
- Exploratory Data Analysis
- Data Preprocessing
- Models Implemented
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance
- Business Insights
- Challenges Faced
- Results
- Folder Structure
- Installation
- How to Run
- Future Enhancements
- Author
- Project Links

---

# 📌 Project Overview

The rapid growth of the smartphone industry has made pricing one of the most critical business decisions for manufacturers and retailers.

This project develops a **Machine Learning Classification Model** capable of predicting the **price range** of a mobile phone using its hardware specifications.

Instead of estimating the exact selling price, the model classifies a phone into one of four price categories:

| Price Range | Class |
|------------|-------|
| Low Cost | 0 |
| Medium Cost | 1 |
| High Cost | 2 |
| Very High Cost | 3 |

The project compares multiple machine learning algorithms, evaluates their performance, identifies the most influential specifications, and recommends the best model for deployment.

---

# 🎯 Business Problem

Pricing a smartphone requires understanding the influence of multiple hardware specifications such as:

- RAM
- Battery Capacity
- Processor
- Internal Storage
- Display Resolution
- Camera Specifications

Incorrect pricing may lead to:

- Reduced sales
- Poor customer satisfaction
- Lower profits
- Incorrect market positioning

A predictive model enables manufacturers to estimate the market segment of a device before launch.

---

# 🎯 Project Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Clean and preprocess the dataset
- Train multiple Machine Learning Classification models
- Compare model performances
- Perform Hyperparameter Optimization
- Identify the most influential mobile specifications
- Generate business insights
- Recommend the best production model

---

# 📊 Dataset Information

Dataset Source

DataMites Capstone Project

Dataset Name

```
datasets_11167_15520_train.csv
```

Dataset Size

- 2000 Mobile Phones
- 20 Input Features
- 1 Target Variable

Target Variable

```
price_range
```

Classes

| Label | Category |
|-------|-----------|
|0|Low Cost|
|1|Medium Cost|
|2|High Cost|
|3|Very High Cost|

---

# 📋 Feature Description

| Feature | Description |
|----------|-------------|
|battery_power|Battery Capacity|
|blue|Bluetooth Availability|
|clock_speed|Processor Speed|
|dual_sim|Dual SIM Support|
|fc|Front Camera|
|four_g|4G Support|
|int_memory|Internal Storage|
|m_dep|Mobile Depth|
|mobile_wt|Phone Weight|
|n_cores|CPU Cores|
|pc|Primary Camera|
|px_height|Screen Resolution Height|
|px_width|Screen Resolution Width|
|ram|RAM|
|sc_h|Screen Height|
|sc_w|Screen Width|
|talk_time|Battery Talk Time|
|three_g|3G Support|
|touch_screen|Touch Screen|
|wifi|WiFi Availability|

---

# 🛠 Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- VS Code

---

# ⚙ Machine Learning Workflow

```
Dataset

      │

      ▼

Data Cleaning

      │

      ▼

Exploratory Data Analysis

      │

      ▼

Feature Scaling

      │

      ▼

Train-Test Split

      │

      ▼

Model Building

      │

      ▼

Hyperparameter Tuning

      │

      ▼

Model Evaluation

      │

      ▼

Feature Importance

      │

      ▼

Business Insights

      │

      ▼

Final Model
```

---

# 📈 Exploratory Data Analysis

EDA includes:

- Dataset Shape
- Data Types
- Missing Values
- Duplicate Records
- Class Distribution
- Correlation Heatmap
- Boxplots
- Histograms
- Pairplots
- Outlier Analysis

---

# 🔄 Data Preprocessing

- Missing Value Check
- Duplicate Removal
- Feature Scaling (StandardScaler)
- Train-Test Split
- Data Validation

---

# 🤖 Machine Learning Models

The following classification algorithms were implemented:

| Model |
|---------|
| Logistic Regression |
| Decision Tree |
| Random Forest |
| K-Nearest Neighbors |
| Support Vector Machine |
| Gradient Boosting |

---

# ⚡ Hyperparameter Tuning

GridSearchCV was applied to optimize the best-performing model.

Optimized Parameters include:

- Regularization
- Tree Depth
- Number of Estimators
- Learning Rate
- Neighbors
- Kernel Parameters

---

# 📊 Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

# 🏆 Model Comparison

| Model | Status |
|--------|--------|
| Logistic Regression | ⭐ Best Model |
| Decision Tree | Evaluated |
| Random Forest | Evaluated |
| KNN | Evaluated |
| SVM | Evaluated |
| Gradient Boosting | Evaluated |

---

# ⭐ Best Model

After comparing all models,

## Logistic Regression

was selected as the final production model because it achieved the highest overall classification performance on unseen data.

---

# 📊 Feature Importance

Random Forest Feature Importance identified the following as the strongest predictors:

🥇 RAM

🥈 Battery Power

🥉 Display Resolution

🏅 Internal Memory

Less Influential Features

- Bluetooth
- WiFi
- Dual SIM

---

# 💼 Business Insights

- RAM is the strongest pricing factor.
- Battery capacity significantly affects the price segment.
- Higher screen resolution increases product value.
- Larger internal memory contributes to premium pricing.
- Connectivity features contribute less because they are standard in most smartphones.

---

# ⚠ Challenges Faced

- Feature scaling required for distance-based models.
- Selecting the best classification algorithm.
- Hyperparameter tuning.
- Comparing multiple evaluation metrics.
- Avoiding overfitting.

---

# 📈 Results

✔ Six machine learning models compared

✔ Logistic Regression achieved the best performance

✔ Hyperparameter tuning improved model robustness

✔ Feature importance successfully identified pricing drivers

✔ Suitable for production deployment

---

# 📁 Project Structure

```
PRCP-1009-CellphonePrice-Prediction/

│

├── Dataset/

│      datasets_11167_15520_train.csv

│

├── Notebook/

│      PRCP_1009_Cellphone_Price_Range_Prediction.ipynb

│

├── Images/

│

├── README.md

│

└── requirements.txt
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/Sameekshagithub/PRCP-1009-CellphonePrice-Prediction-Using-Machine-Learning-Classification-.git
```

Move into the project

```bash
cd PRCP-1009-CellphonePrice-Prediction-Using-Machine-Learning-Classification-
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ How to Run

Open the notebook:

```
PRCP_1009_Cellphone_Price_Range_Prediction.ipynb
```

Run all notebook cells sequentially.

---

# 🚀 Future Enhancements

- Deploy using Streamlit
- Flask API Integration
- Mobile Recommendation System
- Explainable AI using SHAP
- Real-time Mobile Price Prediction
- Deep Learning-based Classification

---

# 👩‍💻 Author

**Sameeksha Rai**

AI Engineer Trainee

Data Science | Machine Learning | Deep Learning | Generative AI

---

# 🔗 Project Links

### GitHub Repository

https://github.com/Sameekshagithub/PRCP-1009-CellphonePrice-Prediction-Using-Machine-Learning-Classification-

---

# ⭐ If you found this project useful

Please consider giving this repository a **⭐ Star** to support the project!

---

<div align="center">

### Thank You ❤️

Made with Python • Scikit-Learn • Data Science • Machine Learning

</div>
