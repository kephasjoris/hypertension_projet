import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
import joblib


def build_preprocessor(df: pd.DataFrame, target_col='Has_Hypertension') -> ColumnTransformer:
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"
    PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

    df = pd.read_csv(DATA_PATH)
    pre = build_preprocessor(df)

    joblib.dump(pre, PREPROCESSOR_PATH)
    print('Preprocessor saved successfully.')