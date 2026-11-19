import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("model.joblib")

FIELDS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def to_frame(data: dict) -> pd.DataFrame:
    row = {
        "Pclass": int(data["Pclass"]),
        "Name": str(data.get("Name") or ""),
        "Sex": str(data["Sex"]).lower(),
        "Age": float(data["Age"]) if data.get("Age") not in (None, "") else None,
        "SibSp": int(data.get("SibSp") or 0),
        "Parch": int(data.get("Parch") or 0),
        "Fare": float(data["Fare"]) if data.get("Fare") not in (None, "") else None,
        "Embarked": data.get("Embarked") or None,
    }
    return pd.DataFrame([row])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    missing = [f for f in ("Pclass", "Sex") if not data.get(f)]
    if missing:
        return jsonify(error=f"Missing required fields: {', '.join(missing)}"), 400
    try:
        X = to_frame(data)
        prob = float(model.predict_proba(X)[0, 1])
    except (ValueError, KeyError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400
    return jsonify(survived=int(prob >= 0.5), probability=round(prob, 4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
