import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# CHEMINS ROBUSTES 🔥
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_best.joblib"
DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"

# LOAD
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# TARGET ENCODING (IMPORTANT)
df['Has_Hypertension'] = df['Has_Hypertension'].map({'No': 0, 'Yes': 1})

X = df.drop(columns=['Has_Hypertension'])
y = df['Has_Hypertension']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# PREDICTIONS
proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)

if isinstance(pred[0], str):
    pred = pd.Series(pred).map({'No': 0, 'Yes': 1})

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC Curve')
plt.legend()
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# CALIBRATION
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], '--')
plt.title('Calibration Curve')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.show()