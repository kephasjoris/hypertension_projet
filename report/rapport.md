# Prédiction de l'Hypertension artérielle — Rapport

**Auteur :** TonNom  
**Date :** YYYY-MM-DD

---

## Résumé
Ce projet présente un pipeline complet pour prédire l'hypertension artérielle à partir d'un jeu de données clinique (≈1985 × 11). Le modèle principal est un **Support Vector Machine (SVM)**. Le rapport couvre : EDA, prétraitement, ingénierie de features, entraînement, évaluation, interprétabilité et déploiement via une API Flask.

---

## 1. Données
- Fichier : `data/hypertension_dataset.csv`  
- Taille : *à remplir après chargement*  
- Colonnes (exemples) : Age, Sex, BMI, SystolicBP, DiastolicBP, Cholesterol, Smoking_Status, Exercise_Level, Family_History, Medication, Has_Hypertension.

---

## 2. Prétraitement
- Imputation : median pour numériques, most_frequent pour catégoriques.  
- Encodage : OneHot pour catégoriques (dans le pipeline), LabelEncoder dans notebooks de démonstration.  
- Normalisation : StandardScaler sur numériques.  
- Pipeline sauvegardé : `models/preprocessor.joblib`.

---

## 3. Ingénierie des features
- `Pulse_Pressure = SystolicBP - DiastolicBP`  
- `BMI_cat` : découpage en bins (`under, normal, over, obese`)  
*Ces transformations sont montrées dans `src/features.py`.*

---

## 4. Modélisation
- Modèle : `sklearn.svm.SVC` (kernel='rbf', probability=True).  
- Validation : split stratifié train/test 80/20.  
- Optimisation : `GridSearchCV` sur `C` et `gamma`, scoring: ROC-AUC.  
- Artefact : `models/svm_best.joblib`.

---

## 5. Résultats (exemple)
> Remplir avec résultats réels après exécution du script.

**Métriques (test set)**  
- ROC-AUC : `0.XXX`  
- Accuracy / Precision / Recall / F1 : voir matrice de classification

**Observations**
- Variables les plus influentes (via SHAP / permutation) : Age, SystolicBP, BMI, Pulse_Pressure.  
- Si le rappel (recall) est prioritaire (clinique), envisager ajuster le seuil ou utiliser class_weight.

---

## 6. Évaluation visuelle
Figures à inclure :
- Histograms des variables clés  
- Matrice de corrélation (numériques)  
- ROC Curve (AUC)  
- Matrice de confusion  
- Calibration plot  
- Precision-Recall curve

(Fichiers générés par `src/evaluate.py` : `report/figures/*.png`)

---

## 7. Interprétabilité
- **SHAP (KernelExplainer)** pour expliquer prédictions individuelles (lent).  
- Alternative plus rapide : permutation importance (sklearn.inspection.permutation_importance).

---

## 8. Déploiement
- API Flask : `src/app.py`, endpoint `/predict` qui accepte JSON des features et renvoie `{pred, proba}`.  
- Dockerfile inclus pour déployer avec Gunicorn.

---

## 9. Limites & perspectives
- Données : vérifier biais, qualité, labels.  
- Améliorations : calibration (Platt / isotonic), comparer XGBoost/LightGBM, gérer imbalance (SMOTE), validation externe.

---

## Annexes
- Code et notebooks (voir `notebooks/`)  
- Commandes : `python src/train.py`, `python src/evaluate.py`, `python src/app.py`.