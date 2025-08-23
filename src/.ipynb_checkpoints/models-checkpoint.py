# src/models.py
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_models_and_params():
    return {
        "RandomForest": (RandomForestClassifier(random_state=42),
                         {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None]}),
        "HistGB": (HistGradientBoostingClassifier(random_state=42),
                   {"clf__max_depth": [3, None], "clf__learning_rate": [0.05, 0.1]}),
        "LogReg": (LogisticRegression(max_iter=500, random_state=42),
                   {"clf__C": [0.1, 1, 10]})
    }

def weighted_metric(cv_acc, f1, recall, auc, w=(0.4, 0.2, 0.2, 0.2)):
    return w[0]*cv_acc + w[1]*f1 + w[2]*recall + w[3]*auc
