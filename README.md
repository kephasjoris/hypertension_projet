# Hypertension Prediction using Machine Learning

A machine learning project for predicting hypertension from patient-related data using Python and Scikit-learn.

## Project Overview

Hypertension is a common health condition that can be influenced by several patient characteristics and lifestyle-related factors.

The objective of this project is to build a machine learning classification pipeline capable of predicting whether a patient is likely to have hypertension.

The project covers the main stages of a practical machine learning workflow:

- Data preparation
- Data preprocessing
- Numerical feature scaling
- Categorical feature encoding
- Machine learning model training
- Hyperparameter optimization
- Model evaluation
- Prediction
- Model deployment preparation

## Dataset

The dataset contains **1,985 observations and 11 variables**.

The target variable is:

```text
Has_Hypertension
```

The target is transformed into a binary classification problem:

```text
No  → 0
Yes → 1
```

The dataset contains both numerical and categorical variables, requiring different preprocessing strategies.

## Machine Learning Approach

The project uses a **Support Vector Machine (SVM)** classifier with an RBF kernel.

The preprocessing and model are combined into a Scikit-learn pipeline.

### Numerical features

Numerical variables are processed using:

- Missing-value imputation with the median
- Standardization using `StandardScaler`

### Categorical features

Categorical variables are processed using:

- Missing-value imputation with the most frequent value
- One-hot encoding
- Unknown categories handled safely with `handle_unknown="ignore"`

### Model

The main classifier is:

```python
SVC(
    probability=True,
    kernel="rbf"
)
```

Hyperparameters are optimized using `GridSearchCV`.

The search includes different values of:

- `C`
- `gamma`

The model evaluation uses a stratified train/test split to preserve the class distribution.

## Model Evaluation

The model is evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The project achieved an ROC-AUC of approximately **0.97** during evaluation.

The confusion matrix obtained during the project was:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 176 | 15 |
| Actual Positive | 25 | 181 |

These results indicate that the trained model was able to distinguish between the two classes effectively on the evaluation set.

> **Important:** This project is an academic machine learning project and is not intended to provide medical diagnosis or replace professional medical advice.

## Project Structure

```text
hypertension_projet/
│
├── data/
│   └── hypertension_dataset.csv
│
├── models/
│   └── svm_best.joblib
│
├── report/
│   └── ...
│
├── src/
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Support Vector Machine (SVM)
- GridSearchCV

### Model Processing

- ColumnTransformer
- Pipeline
- StandardScaler
- OneHotEncoder
- SimpleImputer

### Model Storage

- Joblib

### Deployment

- Flask

## Installation

Clone the repository:

```bash
git clone https://github.com/kephasjoris/hypertension_projet.git
```

Move into the project directory:

```bash
cd hypertension_projet
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The source code is organized inside the `src/` directory.

Depending on the selected script, the workflow can be used to:

1. Load the dataset
2. Prepare the data
3. Train the machine learning pipeline
4. Optimize the model parameters
5. Evaluate the model
6. Save the trained model

The trained model is stored in:

```text
models/svm_best.joblib
```

## API

A Flask API is included as part of the project architecture to make predictions using the trained model.

The prediction endpoint can be used to send patient data to the model and receive a prediction.

Example response:

```json
{
    "prediction": 1
}
```

where:

```text
0 = No hypertension
1 = Hypertension
```

## Skills Demonstrated

This project demonstrates practical skills in:

- Data cleaning
- Data preprocessing
- Exploratory data analysis
- Feature transformation
- Binary classification
- Support Vector Machines
- Hyperparameter tuning
- Model evaluation
- Model serialization
- API development
- Python development

## Author

**Ganhoume Kephas Joris**

Junior Data Analyst | Python | SQL | Data Cleaning | Machine Learning

GitHub:  
https://github.com/kephasjoris

---

## Disclaimer

This project was developed for educational and portfolio purposes.

It should not be used as a medical diagnostic system or as a substitute for professional medical advice.
