import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from data_preprocessing import build_preprocessor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def train(save_path='hypertension_projectprojet/models'):
    
    DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"
    
    df = pd.read_csv(DATA_PATH)
    target = 'Has_Hypertension'
    X = df.drop(columns=[target])
    y = df[target].map({'No': 0, 'Yes': 1})

    preprocessor = build_preprocessor(df, target_col=target)

    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', SVC(probability=True))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    param_grid = {
        'clf__C': [0.1, 1, 10, 50],
        'clf__gamma': ['scale', 0.01, 0.001],
        'clf__kernel': ['rbf']
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print('Best params:', grid.best_params_)

    # prediction
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print('ROC AUC:', roc_auc_score(y_test, y_proba))

    # ... après avoir construit X (DataFrame) et entraîné best (GridSearch best_estimator_)
    feature_cols = X.columns.tolist()
    joblib.dump(feature_cols, f"{save_path}/feature_cols.joblib")

    # calculer valeurs par défaut (numériques -> mediane, catégoriques -> mode)
    defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            defaults[col] = X[col].mode().iloc[0] if not X[col].mode().empty else ""
    joblib.dump(defaults, f"{save_path}/defaults.joblib")


    # save model
    joblib.dump(best, f"{save_path}/svm_best.joblib")
    print('Model saved to', f"{save_path}/svm_best.joblib")


if __name__ == '__main__':
    train()