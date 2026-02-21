# ML Syntax Reference Guide: California Housing Project

This document serves as a comprehensive reference for the libraries, methods, and syntax used in the California Housing ML project. It focuses on practical use cases and engineering implementation.

---

## 1. Data Acquisition & OS Utilities

### `os` Module
Standard library for interacting with the operating system and file paths.
- `os.path.join(path, *paths)`: Concatenates path components intelligently using the correct separator for the OS.
    - *Use Case*: Constructing file paths for datasets (e.g., `datasets/housing/housing.csv`).
- `os.makedirs(name, exist_ok=True)`: Creates a directory and any necessary intermediate directories. `exist_ok=True` prevents errors if the directory exists.
    - *Use Case*: Ensuring the dataset folder exists before downloading.

### `urllib.request` & `tarfile`
- `urllib.request.urlretrieve(url, filename)`: Downloads a remote resource to a local file.
- `tarfile.open(name)`: Opens a tar archive; used with `.extractall()` to unzip data.

### `zlib` & Data Hashing
- `zlib.crc32(data)`: Computes a CRC32 checksum.
    - *Use Case*: Creating stable train/test splits based on row identifiers or geographic coordinates. The `& 0xffffffff` bitwise operation is used to ensure consistency across different Python versions/platforms.

---

## 2. Data Manipulation (Pandas)

### Data Loading & Inspection
- `pd.read_csv(filepath)`: Loads a CSV file into a DataFrame.
- `df.head(n)`: Returns the first `n` rows (default 5). Quick visual check of data.
- `df.info()`: Provides a concise summary of the DataFrame, including non-null counts and memory usage.
    - *Use Case*: Identifying columns with missing values (e.g., `total_bedrooms`).
- `df.describe()`: Generates descriptive statistics for numerical columns (mean, std, min, max, quartiles).
- `df.value_counts()`: Returns counts of unique values in a Series.
    - *Use Case*: Analyzing categorical features like `ocean_proximity`.

### Data Cleaning & Selection
- `df.drop(labels, axis=1)`: Removes columns (axis=1) or rows.
    - *Use Case*: Separating features from labels (dropping the target column).
- `df.dropna(subset=[cols])`: Removes rows with missing values in specified columns.
- `df.fillna(value)`: Fills NA/NaN values with a specified value (e.g., the median).
- `df.isnull().sum()`: Returns the number of missing values per column.
- `df.iloc[indices]`: Purely integer-location based indexing.
- `df.loc[indices/names]`: Access a group of rows and columns by label(s) or a boolean array.
- `df.reset_index()`: Resets the index to the default integer index and moves the old index into a column.
    - *Use Case*: Turning row indices into a unique identifier column for stable splitting.
- `df.corr(numeric_only=True)`: Computes pairwise correlation of columns. `numeric_only=True` is vital when the dataset contains non-numeric (e.g., categorical) columns.

### Binning & Transformation
- `pd.cut(x, bins, labels)`: Segments and sorts data values into bins.
    - *Use Case*: Creating the `income_cat` attribute for stratified sampling.

---

## 3. Numerical Operations (NumPy)

- `np.random.permutation(n)`: Randomly permutes a sequence or range.
    - *Use Case*: Shuffling indices for custom train/test splitting.
- `np.random.seed(42)`: Sets the seed for the random number generator to ensure reproducibility.
- `np.linspace(start, stop, num)`: Returns evenly spaced numbers over a specified interval.
- `np.inf`: Representation of infinity. Used as an upper bound in `pd.cut`.
- `np.c_[array1, array2]`: Translates slice objects to concatenation along the second axis (column-wise).
    - *Use Case*: Combining original features with derived features in custom transformers.

---

## 4. Visualization (Matplotlib & Pandas Plotting)

- `df.hist(bins, figsize)`: Plots histograms of all numerical columns.
- `df.plot(kind='scatter', x, y, ...)`: Versatile plotting function.
    - *Use Case*: Visualizing geographical data (`latitude`/`longitude`) and price correlations.
- `alpha`: Parameter for point transparency (helps visualize density).
- `s`: Sizes of points (e.g., linked to `population`).
- `c`: Color of points (e.g., linked to `median_house_value`).
- `cmap`: Colormap (e.g., `plt.get_cmap("jet")`).
- `pandas.plotting.scatter_matrix(df)`: Plots a matrix of scatter plots for multiple attributes.
    - *Use Case*: Identifying non-linear correlations between features.

---

## 5. Machine Learning Essentials (Scikit-Learn)

### Model Selection & Sampling
- `train_test_split(data, test_size, random_state)`: Standard random splitting utility.
- `StratifiedShuffleSplit(n_splits, test_size, random_state)`: Provides train/test indices to split data in stratified fashion.
    - *Use Case*: Ensuring the test set matches the distribution of `income_cat`.

### Data Preprocessing
- `SimpleImputer(strategy)`: Transformer for completing missing values (e.g., using "median").
    - `.fit(X)`: Computes the median for each column.
    - `.transform(X)`: Replaces missing values with the learned medians.
- `OrdinalEncoder()`: Encodes categorical features as an integer array (0, 1, 2...).
- `OneHotEncoder()`: Encodes categorical features as a one-hot (binary) numeric array.
    - *Use Case*: Handling `ocean_proximity`.
- `StandardScaler()`: Standardizes features by removing the mean and scaling to unit variance.

### Pipelines & Composition
- `Pipeline(steps)`: Sequences multiple transformers and an optional final estimator.
    - *Use Case*: Chaining imputer → custom adder → scaler.
- `ColumnTransformer(transformers)`: Applies different transformers to different columns (e.g., numerical vs. categorical).
    - *Use Case*: The "Full Pipeline" that handles the entire dataset transformation in one call.

### Custom Transformers
- `BaseEstimator` & `TransformerMixin`: Base classes for custom transformers.
    - *Necessity*: `TransformerMixin` adds `fit_transform()` automatically. `BaseEstimator` provides `get_params()` and `set_params()` for grid search.

### Estimator Inspection Attributes
SciKit-Learn estimators store learned parameters with a trailing underscore.
- `imputer.statistics_`: Stores the computed medians for each attribute.
- `encoder.categories_`: Stores the list of categories found during fitting.
- `model.feature_importances_`: Stores the importance/weight of each feature.
- `grid_search.best_params_`: Stores the optimal hyperparameter combination.
- `grid_search.best_estimator_`: The actual model instance with the best hyperparameters.

---

## Scikit-Learn: Model Training & Evaluation

### `sklearn.linear_model`
- `LinearRegression()`: Ordinary least squares Linear Regression.
- `model.fit(X, y)`: Trains the model.
- `model.predict(X)`: Makes predictions on new data.

### `sklearn.tree`
- `DecisionTreeRegressor()`: A model that predicts by learning simple decision rules inferred from the data features.

### `sklearn.ensemble`
- `RandomForestRegressor()`: A meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting.

### `sklearn.metrics`
- `mean_squared_error(y_true, y_pred)`: Measures the average squared difference between estimated and actual values.

### `sklearn.model_selection`
- `cross_val_score(estimator, X, y, scoring, cv)`: Evaluates a score by cross-validation. Note that `scoring="neg_mean_squared_error"` returns negative values (utility vs cost convention).
- `GridSearchCV(estimator, param_grid, cv, scoring)`: Exhaustive search over specified parameter values for an estimator.

---

## Scipy: Statistical Analysis

### `scipy.stats`
- `stats.t.interval(confidence, df, loc, scale)`: Computes a confidence interval based on a t-distribution. Used in this project to estimate the 95% confidence interval of the prediction error.
