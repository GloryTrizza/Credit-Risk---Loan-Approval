# Credit-Risk---Loan-Approval

This project focuses on building and evaluating a **credit risk loan approval model**. The goal is to predict the likelihood of loan default based on various features such as income, employment length, age, and loan amount using machine learning classification models.

## Project Overview

This project utilizes machine learning techniques, particularly focusing on **Random Forest Classifier** and **LightGBM**, to predict loan status (approved/denied). The workflow includes data cleaning, imputation, feature engineering, model training, and evaluation.

### Key Features:
- **Dataset**: Credit risk dataset with features like `person_age`, `person_income`, `loan_amount`, and others.
- **Pipeline**: A machine learning pipeline is created using scikit-learn’s `Pipeline` and `ColumnTransformer` for efficient data preprocessing and model training.
- **Imputation**: Missing numerical values are handled using `IterativeImputer`.
- **Hyperparameter Tuning**: Models are tuned using `RandomizedSearchCV`.
- **Evaluation**: Confusion matrix, precision-recall curve, and learning curves are used to evaluate model performance.

## Installation

Clone the repository and install the necessary packages:



### Main Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `dask[dataframe]`

### Dataset
The dataset used in this project is stored in `credit_risk_dataset.csv`. It contains the following columns:

- **person_age**: The age of the loan applicant.
- **person_income**: The applicant's income.
- **loan_amnt**: The amount of the loan.
- **loan_int_rate**: The interest rate of the loan.
- **loan_status**: The target variable indicating loan approval (0: Denied, 1: Approved).

### Project Workflow

1. **Data Preprocessing**  
   - **Removing Duplicates**: The dataset is cleaned of duplicate entries.  
   - **Handling Missing Data**: Numerical missing values are imputed using an `IterativeImputer` (with options like `LinearRegression`).  
   - **Categorical Encoding**: Categorical features are encoded using `OneHotEncoder` for machine learning compatibility.

2. **Model Building**  
   - **Pipeline Setup**: A pipeline is created using `ColumnTransformer` and `RandomForestClassifier`.  
   - **Hyperparameter Tuning**: `RandomizedSearchCV` is used to find the best hyperparameters for the model.

3. **Model Evaluation**  
   - **Confusion Matrix**: Used to assess classification performance.  
   - **Classification Report**: Precision, recall, and F1-score are evaluated.  
   - **Learning Curves**: Training and validation accuracy are plotted to understand model performance.  
   - **Precision-Recall Curves**: Used to visualize the trade-off between precision and recall.

4. **Model Performance**  
   The best-performing model is evaluated on a test dataset. A confusion matrix and classification report provide insights into the model’s accuracy and classification metrics.

### Usage
After setting up the environment and installing the dependencies, you can run the project using:

### Evaluation
The model evaluation process includes:

- **Confusion Matrix**: Displays the performance of the classification model.
- **Classification Report**: Shows precision, recall, F1-score, and accuracy for each class.
- **Precision-Recall Curve**: Plots the trade-off between precision and recall for the model.
- **Learning Curves**: Visualizes the training and validation accuracy to assess model overfitting or underfitting.

### Future Improvements
Potential improvements to the model include:

- Incorporating additional features to better capture the applicant’s financial profile.
- Exploring advanced machine learning models like **XGBoost** or **Neural Networks**.
- Implementing a more robust hyperparameter tuning strategy with **Bayesian Optimization**.
