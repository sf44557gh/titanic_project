# =====================================
# src/preprocessing.py
# =====================================

import pandas as pd

def clean_df(df):
    """
    Nettoyage et création de nouvelles features pour le dataset Titanic.
    """
    df = df.copy()
    
    # Taille de la famille
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    # IsAlone : personne seule à bord
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    # Extraction du titre depuis le nom
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
    
    return df

def get_features(df, use_isalone=True, use_cabin=False):
    """
    Retourne la liste des features à utiliser selon les flags.
    """
    base = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
    if use_isalone: 
        base.append("IsAlone")
    if use_cabin and "Cabin" in df.columns: 
        base.append("Cabin")
    return base

def make_preprocessor(df, features):
    """
    Création du ColumnTransformer pour le préprocessing : imputation + scaling + one-hot.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    
    categorical = [f for f in features if df[f].dtype == "object"]
    numerical = [f for f in features if df[f].dtype != "object"]
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),
        
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ])
    
    return preprocessor
