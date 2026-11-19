import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from features import engineer

df = pd.read_csv("data/train.csv")
X, y = df.drop(columns="Survived"), df["Survived"]

num = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]
cat = ["Sex", "Embarked", "Title"]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ]), cat),
])

pipe = Pipeline([
    ("fe", FunctionTransformer(engineer)),
    ("pre", pre),
    ("clf", RandomForestClassifier(random_state=42)),
])

grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [5, 7, None],
    "clf__min_samples_leaf": [1, 3, 5],
}

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
search = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
search.fit(X_tr, y_tr)
print("Best params:", search.best_params_)
print(f"CV accuracy:      {search.best_score_:.3f}")
print(f"Hold-out accuracy: {search.score(X_val, y_val):.3f}")

# Refit the best config on all labelled data, then save
final = search.best_estimator_.fit(X, y)
joblib.dump(final, "model.joblib")
print("Saved model.joblib")

# Predictions for test.csv (Kaggle-style submission)
test = pd.read_csv("data/test.csv")
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final.predict(test)}).to_csv(
    "predictions.csv", index=False
)
print("Saved predictions.csv")
