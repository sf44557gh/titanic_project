# src/preprocessing.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def clean_df(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
    return df

def get_features(df, use_isalone=True, use_cabin=False):
    base = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
    if use_isalone: base.append("IsAlone")
    if use_cabin and "Cabin" in df.columns: base.append("Cabin")
    return base

def make_preprocessor(df, features, strategy_num="median"):
    categorical = [f for f in features if df[f].dtype == "object"]
    numerical = [f for f in features if df[f].dtype != "object"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy=strategy_num)),
            ("scaler", StandardScaler())
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ])
    return preprocessor
