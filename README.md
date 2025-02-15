# Diabetes Prediction

## Dataset Description

The dataset `diabetes.csv` contains 768 rows and 9 columns, with relevant features such as `Insulin`, `BMI`, `Glucose`, `BloodPressure`, etc. Upon inspecting the dataset using the `.describe()` function, it was observed that some columns of interest contain values of `0`, which can negatively impact the performance of classification models.

## Motivation

The goal of this project is to apply supervised learning methods to classify whether a person is diabetic based on given medical data.

## Objectives

1. Handle and explore a multi-column dataset to understand the distribution of data.
2. Apply dimensionality reduction techniques (PCA) to evaluate their utility and impact on model performance.
3. Implement and compare multiple supervised learning algorithms to analyze their performance.

## Dataset Processing

### Initial Analysis

The dataset was imported and analyzed using the following methods:
- `.head()` – to display the first few rows.
- `.describe()` – to get statistical insights.
- `.corr()` – to examine feature correlations.

### Handling Missing Data

It was observed that columns such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contained values of `0`, which are not plausible in a medical context. These values were treated as missing values and replaced with the mean of their respective columns.

## Exploratory Data Analysis

A `pairplot` was generated to visualize the relationship between key features: `Glucose`, `Insulin`, `BMI`, and `Age`. The plot revealed clusters in the data with few anomalies.

## Data Preprocessing

Due to the varying scales of the features, `MinMaxScaler()` was applied to normalize the data. This is particularly important for distance-based algorithms such as K-Nearest Neighbors (KNN) and also benefits other models like Support Vector Machines (SVM) and Logistic Regression.

## Principal Component Analysis (PCA)

PCA was applied to reduce the dimensionality of the dataset and observe its impact on model performance. The `apply_pca()` function was created to handle this transformation.

## Classification Models

The following supervised learning models were implemented with and without PCA:

### Support Vector Machine (SVM)
- Without PCA: `svm_without_pca()`
- With PCA: `svm_with_pca()`

### Gaussian Naive Bayes (GNB)
- Without PCA: `gnb_without_pca()`
- With PCA: `gnb_with_pca()`

### K-Nearest Neighbors (KNN)
- Without PCA: `knn_without_pca()`
- With PCA: `knn_with_pca()`

### Random Forest (RF)
- Without PCA: `rf_without_pca()`
- With PCA: `rf_with_pca()`

## Model Evaluation

Each model was trained and evaluated as follows:
1. Split the data into training and testing sets.
2. Train the model using appropriate hyperparameters.
3. Obtain predictions and calculate accuracy.
4. Display a classification report.
5. Plot a confusion matrix using `plot_confusion_matrix()`.

For KNN and RF, hyperparameter tuning was performed by iterating over different values for the number of neighbors and estimators, respectively.

## Results

Performance evaluations and detailed outputs can be found in the Jupyter Notebook included in this repository.

## Conclusions

Final conclusions and additional details are provided in the Jupyter Notebook.

## Requirements

The following libraries are required to run this project:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
```

2. Navigate to the project directory:
```bash
cd diabetes-prediction
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook Diabetes_Prediction.ipynb
```

## License

This project is licensed under the MIT License.

