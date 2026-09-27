import joblib
import shap
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_best.joblib"
DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['Has_Hypertension'])

# FONCTION SHAP
def f(x):
    return model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1]

# SAMPLE (SUR DONNÉES BRUTES)
X_sample = shap.sample(X, 50)

explainer = shap.KernelExplainer(f, X_sample)

shap_values = explainer.shap_values(X.iloc[:50])

shap.summary_plot(shap_values, X.iloc[:50])