# California Housing: Technical ML Engineering Documentation

This document provides a technical walkthrough of the California Housing price prediction project, following the end-to-end Machine Learning lifecycle. It focuses on the engineering rationale, theoretical foundations, and implementation details.

---

## 1. Problem Framing and Business Objective

### 1.1 Business Objective
The primary goal is to predict the median housing price in any California district, given several metrics. This model's output will be fed into another ML system to determine whether it is worth investing in a given area.

### 1.2 ML Task Framing
> [!NOTE]
> **Supervised vs. Unsupervised Learning**
> - **Supervised**: The algorithm is trained on a dataset that includes labels (the desired solutions). In this case, the `median_house_value`.
> - **Unsupervised**: The dataset is unlabeled, and the goal is usually to find hidden patterns or structures (e.g., clustering).

The project is framed as:
- **Supervised Learning**: We have labeled training examples.
- **Regression Task**: We are predicting a continuous numerical value.
- **Univariate Multiple Regression**: We use multiple features (longitude, latitude, income, etc.) to predict a single value (price).
- **Batch Learning**: The system does not need to learn on the fly; it can be trained on a fixed dataset.

### 1.3 Performance Measure: RMSE vs. MAE
> [!IMPORTANT]
> **L2 vs. L1 Norm Intuition**
> - **RMSE (Root Mean Square Error)**: Corresponds to the **Euclidean norm ($L_2$)**. It gives higher weight to large errors (outliers).
> - **MAE (Mean Absolute Error)**: Corresponds to the **Manhattan norm ($L_1$)**. It is more robust to outliers.

For this project, **RMSE** is generally preferred unless there are many significant outliers, as it penalizes large errors more heavily, which is desirable for price predictions.

---

## 2. Data Acquisition and Inspection

### 2.1 Automated Data Fetching
To ensure reproducibility and handle potential data updates, we implement a fetching script.

```python
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path, filter="data")
```

### 2.2 Initial Inspection
Primary inspection focuses on data types, missing values, and distributions.

```python
import pandas as pd

housing = pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))
housing.info()             # Check for nulls and Dtypes
housing.describe()         # Statistical summary
housing.hist(bins=50)      # Visualizing distributions
```

**Key Findings:**
- `total_bedrooms` has missing values (requires imputation).
- `ocean_proximity` is a categorical attribute.
- Targets are capped at $500,000.
- Features are on vastly different scales.
- Many distributions are "tail-heavy."

---

## 3. Test Set Design and Sampling Strategy

### 3.1 Avoiding Data Snooping Bias
> [!CAUTION]
> **Data Snooping Bias**
> If you look at the test set before training, you might spot patterns that lead you to choose a specific model. This results in optimistic performance estimates that won't generalize.

### 3.2 Stratified Sampling Strategy
A purely random split might introduce sampling bias, especially if the dataset is small or if certain features are highly predictive. `median_income` is a critical predictor, so we use **Stratified Sampling** to ensure the test set is representative of the various income categories.

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Binning numerical income into categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

**Reasoning**: This maintains the ratio of income levels in both sets, minimizing the likelihood that the model's performance varies significantly between development and production due to distribution shifts.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Geographical Visualization
Visualizing longitude/latitude reveals clusters (densities) and correlation with price.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
```

### 4.2 Correlation Analysis
> [!WARNING]
> **Correlation Limitations**
> The correlation coefficient only measures **linear** relationships ($y = mx + b$). It misses non-linear dependencies.

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

### 4.3 Feature Engineering Intuition
Existing attributes may not be sufficient. Creating derived features can often provide more signal to the model.
- **Rooms per household**: Is the house large relative to the number of occupants?
- **Bedrooms per room**: What is the bedroom density?

---

## 5. Data Preprocessing Pipelines

### 5.1 Data Cleaning (Imputation)
We use `SimpleImputer` to fill missing values. 

> [!IMPORTANT]
> **Pipeline Discipline: fit() only on Training Data**
> Never use the test set statistics (mean, median) to fill training data. Fit the imputer/scaler on the training set and use that learned state to `transform()` both sets.

### 5.2 Handling Text and Categorical Attributes
`ocean_proximity` is converted via **One-Hot Encoding**. This prevents the model from assuming a numerical order (as with Ordinal Encoding) where none exists.

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# Returns a sparse matrix; use toarray() for visualization
housing_cat_1hot.toarray()
```

> [!NOTE]
> **Sparse vs. Dense Matrices**
> One-hot encoding creates many columns. SciKit-Learn returns a **Sparse Matrix** (storing only non-zero indices) to save memory. Use `toarray()` only if necessary.

### 5.3 Custom Transformers
We encapsulate feature engineering into a class for inclusion in pipelines. This ensures that the same transformations are applied to training, test, and production data.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # Nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]
```

### 5.4 Feature Scaling
> [!TIP]
> **Scaling impact on Optimization**
> Many algorithms (Gradient Descent, SVMs, NN) converge much faster or perform better when features are on the same scale.

- **Standardization (`StandardScaler`)**: Subtract mean, divide by SD. Less sensitive to outliers.

### 5.5 The `ColumnTransformer` Pipeline
Combining all steps into a single, modular pipeline ensures consistency and prevents data leakage.

```python
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
```

---

---

## 6. Select and Train a Model

### 6.1 Training and Evaluating
We start with a **Linear Regression** model as a baseline. However, as Géron notes, a simple model might underfit the data.

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

To explore more complex patterns, we try a `DecisionTreeRegressor`. Interestingly, this model yields an RMSE of **0.0** on the training set, indicating severe **overfitting** (it has simply memorized the data).

### 6.2 Better Evaluation Using Cross-Validation
To get a more realistic estimate of model performance, we use **K-fold Cross-Validation** (via `cross_val_score`). This splits the training set into 10 subsets (folds) and trains/evaluates the model 10 times.

- **Linear Regression RMSE (Mean)**: ~69,104
- **Decision Tree RMSE (Mean)**: ~71,432

Despite the initial perfect (0.0) score, the Decision Tree actually performs worse than the Linear model when cross-validated. This highlights the importance of not trusting training error alone.

### 6.3 Ensemble Learning: Random Forest
Finally, we try a `RandomForestRegressor`, which trains many Decision Trees on random subsets of the features and then averages their predictions. This ensemble method typically performs much better and is less prone to overfitting than a single tree.

---

## 7. Fine-Tune Your Model

### 7.1 Grid Search
Instead of fiddling with hyperparameters manually, we use `GridSearchCV`. We provide a set of hyperparameters (e.g., `n_estimators`, `max_features`) and Scikit-Learn uses cross-validation to find the best combination.

- **Best Parameters found**: `max_features=6, n_estimators=30`
- **Feature Importance**: Analyzing the best model reveals which features are most predictive. In this project, `median_income` is by far the most important feature, followed by the `INLAND` category and `pop_per_hhold`.

### 7.2 Final Evaluation on the Test Set
After selecting the best model and tuning its hyperparameters, we evaluate it on the **test set**.

> [!IMPORTANT]
> **Data Leakage Check**: Call `transform()`, not `fit_transform()`, on the test set. This ensures you use the parameters learned from the training data, accurately reflecting how the model will perform on unseen data.

- **Final RMSE**: ~47,491
- **95% Confidence Interval**: ~45,481 to ~49,419

This interval gives a measure of how precise the prediction is. A relatively narrow interval (in terms of housing prices) suggests the model is fairly stable.

---

## Key Concepts Recap
- **Data Snooping**: The silent killer of ML models. Maintain strict separation of test data.
- **Stratified Sampling**: Essential for biased/small datasets to ensure representativeness.
- **Pipelines**: The foundation for production-ready, reproducible code.
- **Bias–Variance Trade-off**: Finding the balance between underfitting and overfitting.
- **Cross-Validation**: A robust way to evaluate models without leaking data.

## Design Decisions Explained
1. **Decision**: Using `StratifiedShuffleSplit` instead of `train_test_split`.
   - **Reason**: Income is a heavy predictor; splitting randomly could lead to a test set unrepresentative of high/low-income districts.
2. **Decision**: One-Hot Encoding for `ocean_proximity`.
   - **Reason**: The proximity values are nominal; ordinal encoding would imply an artificial ranking that the model might try to exploit.
3. **Decision**: Custom Transformer for attribute addition.
   - **Reason**: Automates exploration of feature combinations during GridSearchCV and Cross-Validation.
4. **Decision**: Using `GridSearchCV` for hyperparameter tuning.
   - **Reason**: More systematic and reproducible than manual "trial and error."
