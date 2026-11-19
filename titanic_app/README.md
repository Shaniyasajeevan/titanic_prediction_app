# Titanic Survival Predictor (Flask only)

    pip install -r requirements.txt
    python train.py      # trains and saves model.joblib
    python app.py        # serves at http://localhost:5000

API: POST /predict with JSON {Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked}
