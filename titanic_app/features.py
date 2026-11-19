import pandas as pd


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw Titanic columns into model features. Used for both training and serving."""
    df = df.copy()
    df["Title"] = (
        df["Name"].fillna("").str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    )
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    common = {"Mr", "Mrs", "Miss", "Master"}
    df["Title"] = df["Title"].where(df["Title"].isin(common), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df[["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone"]]
