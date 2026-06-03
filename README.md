# Missing Category Imputation: Constant Category-Based Missing Value Recovery

[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Preprocessing-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Strategy-Missing%20Category%20Imputation-orange)](https://scikit-learn.org/stable/modules/impute.html)
[![Dataset](https://img.shields.io/badge/Dataset-Housing%20Dataset-green)](./housing_dataset.csv)

---

## 🏗️ Project Overview

Missing values in categorical features are a common challenge in real-world datasets. While techniques such as Frequent Value Imputation replace missing values with the most common category, doing so may hide potentially useful information carried by the missing values themselves.

This project explores **Missing Category Imputation**, a preprocessing technique that replaces missing values with a new artificial category such as **"Missing"**, **"Unknown"**, or **"Not Available"**.

Using the **Housing Dataset (`housing_dataset.csv`)**, this notebook demonstrates how to preserve missingness information while preparing categorical features for machine learning models using Scikit-Learn's **`SimpleImputer(strategy='constant')`**.

---

## 🛠️ Advanced Engineering Mechanics

### 1. What is Missing Category Imputation?

Missing Category Imputation creates a completely new category specifically for missing observations.

Example:

| Neighborhood |
| ------------ |
| Urban        |
| Rural        |
| Missing      |
| Urban        |
| Missing      |

After Imputation:

| Neighborhood |
| ------------ |
| Urban        |
| Rural        |
| Missing      |
| Urban        |
| Missing      |

Instead of guessing what the missing value should be, we explicitly inform the model that the value was unavailable.

---

### 2. Why Use Missing Category Imputation?

Unlike Mean, Median, or Mode Imputation, this approach:

* Preserves missing-value information
* Avoids introducing statistical bias
* Works naturally with categorical variables
* Helps tree-based algorithms learn patterns from missingness itself

This method is especially useful when the absence of information may carry predictive value.

---

### 3. The Preprocessing Pipeline Architecture

```text
                 ┌─────────────────────────────────────┐
                 │       Raw Housing Dataset           │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │      Missing Value Detection        │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │  Missing Category Generation        │
                 └─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                                                     │
        ▼                                                     ▼

 Missing → "Missing"                           Missing → "Unknown"

        │                                                     │
        ▼                                                     ▼

 Category Preservation                    Explicit Missing Signal

        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               ▼

                 ┌─────────────────────────────────────┐
                 │     Complete Feature Matrix         │
                 └─────────────────────────────────────┘
```

---

## 🔬 Implementation Workflow

The notebook follows a complete missing-value handling workflow:

### Step 1: Dataset Exploration

* Load Housing Dataset
* Identify missing values
* Analyze categorical columns

### Step 2: Missing Value Profiling

* Calculate null percentages
* Determine affected features
* Visualize missing-value patterns

### Step 3: Manual Category Creation

Example:

```python
df['column'] = df['column'].fillna('Missing')
```

This creates an explicit category for missing observations.

### Step 4: Scikit-Learn Implementation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy='constant',
    fill_value='Missing'
)
```

This approach integrates seamlessly into production ML pipelines.

### Step 5: Distribution Validation

* Compare category frequencies before imputation
* Compare category frequencies after imputation
* Analyze the newly created category

---

## 📊 Missing Value Strategy Comparison

| Strategy                    | Data Type   | Information Preservation | Bias Risk | Best Use Case                  |
| --------------------------- | ----------- | ------------------------ | --------- | ------------------------------ |
| Mean Imputation             | Numerical   | Low                      | Medium    | Symmetric Numerical Data       |
| Median Imputation           | Numerical   | Low                      | Low       | Skewed Numerical Data          |
| Frequent Value Imputation   | Categorical | Medium                   | Medium    | Low Missingness                |
| Missing Category Imputation | Categorical | High                     | Low       | Missingness May Be Informative |

---

## ⚠️ Engineering Limitations

### Increased Cardinality

A new category is added to the feature, increasing the total number of categories.

### Sparse Categories

The new category may contain very few observations.

### Potential Noise

If missing values are completely random, the new category may not contribute meaningful information.

### Encoding Considerations

Additional categories may increase dimensionality after One-Hot Encoding.

---

## 🚀 Why This Technique Matters

Missing Category Imputation is widely used because it:

* Preserves missingness information
* Prevents information leakage
* Works well with categorical features
* Integrates easily with Scikit-Learn pipelines
* Supports production-grade ML workflows

It is particularly valuable when missing values themselves may contain predictive signals.

---

## 💻 Tech Stack & Dependencies

### Programming Language

* Python 3.9+

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn (`SimpleImputer`)

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook

---

## 📚 Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 🎯 Key Learning Outcomes

After completing this notebook, you will understand:

* What Missing Category Imputation is
* When to use constant category imputation
* How to create custom missing categories
* How to implement `SimpleImputer(strategy='constant')`
* The advantages and limitations of this technique
* How missingness can become a predictive feature

---

## 🏁 Conclusion

Missing Category Imputation is one of the most effective preprocessing techniques for handling missing categorical data when missingness itself may carry useful information.

Rather than replacing missing values with existing categories, this approach preserves the absence of information by creating a dedicated category. As a result, machine learning models can learn patterns associated with missing data while maintaining dataset integrity.

This notebook demonstrates both the theoretical foundations and practical implementation of Missing Category Imputation using the Housing Dataset, providing a strong foundation for production-ready preprocessing pipelines.

---

## 🚀 Future Improvements

* Compare Missing Category vs Frequent Value Imputation
* Evaluate model performance impact
* Integrate with One-Hot Encoding pipelines
* Explore advanced categorical encoding techniques
* Build complete preprocessing workflows using ColumnTransformer
