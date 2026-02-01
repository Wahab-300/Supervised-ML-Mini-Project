## Machine Learning Mini Project 

A comprehensive collection of supervised and unsupervised machine learning projects demonstrating various algorithms and techniques including Linear Regression, K-Means Clustering, and Principal Component Analysis (PCA).

## What's Inside? 

This repository has 3 projects:

1. **Linear Regression** - Predict student scores from study hours
2. **K-Means Clustering** - Group similar data together
3. **PCA** - Reduce data dimensions

## Requirements 

You need Python and these libraries:
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation 

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

## Projects 

### 1. Linear Regression (Supervised Learning)

**Goal:** Predict exam scores based on hours studied

**File:** `SupervisedLearning_small_project_.ipynb`

**What it does:**
- Takes hours studied as input
- Predicts exam score as output
- Example: 1 hour = 52 marks, 5 hours = 75 marks

**Results:**
- Error Rate: Very low (RMSE: 0.79)
- Accuracy: Good predictions

---

### 2. K-Means Clustering (Unsupervised Learning)

**Goal:** Find groups in data automatically

**File:** `K-Means.ipynb`

**What it does:**
- Groups similar data points together
- No labels needed
- Finds patterns automatically

---

### 3. PCA (Unsupervised Learning)

**Goal:** Make data smaller but keep important info

**File:** `PCA_Unsupervised_Learning_.ipynb`

**What it does:**
- Reduces data from many columns to 2 columns
- Keeps 99.65% of important information
- Makes data easier to visualize

**Results:**
- Component 1: 99.65% variance
- Component 2: 0.35% variance

## Dataset 

**File:** `Supervised_L(small Project).csv`

Simple dataset with 6 rows:

| Hours | Scores |
|-------|--------|
| 1     | 52     |
| 2     | 57     |
| 3     | 65     |
| 4     | 70     |
| 5     | 75     |
| 6     | 80     |

## How to Run 

1. Download or clone this repository
2. Open Jupyter Notebook
3. Open any `.ipynb` file
4. Click "Run All" or run cells one by one

## Quick Start Example 
 
```python
# Load data
import pandas as pd
data = pd.read_csv("Supervised_L(small Project).csv")

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[['Hours']], data[' Scores'])

# Predict
score = model.predict([[3]])  # 3 hours of study
print(f"Predicted score: {score[0]:.0f}")
```

## What You'll Learn 

- How to predict values (Regression)
- How to find groups in data (Clustering)
- How to reduce data size (PCA)
- How to evaluate model performance
- Basic data analysis with Python

## Technologies 

- Python 3.13
- Pandas - Data handling
- NumPy - Math operations
- Scikit-learn - Machine learning
- Matplotlib - Graphs

## Author 

Your Name  
GitHub: https://github.com/Wahab-300

## Questions? 
Feel free to open an issue or contact me.
wahabcodes1@gmail.com


