"""
API FastAPI pour la prediction de churn bancaire.
Base sur CatBoost + Features Engineering.
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent

app = FastAPI(
    title="Churn Prediction API",
    description="Prediction du risque de desabonnement client (BNP Paribas)",
    version="2.0"
)

try:
    model = joblib.load(BASE_DIR / "churn_model_catboost.pkl")
    preprocessor = joblib.load(BASE_DIR / "churn_preprocessor.pkl")
    threshold = joblib.load(BASE_DIR / "churn_threshold.pkl")
    metadata = joblib.load(BASE_DIR / "churn_metadata.pkl")
    print(f"OK Modele charge : {metadata['model_name']}")
    print(f"OK Seuil          : {threshold}")
    print(f"OK F1 optimal     : {metadata['f1_optimal']}")
except Exception as e:
    print(f"ERREUR chargement : {e}")
    raise


class Client(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 600, "Geography": "France", "Gender": "Female",
                "Age": 50, "Tenure": 5, "Balance": 100000,
                "NumOfProducts": 3, "HasCrCard": 1, "IsActiveMember": 0,
                "EstimatedSalary": 80000
            }
        }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Age_At_Join'] = df['Age'] - df['Tenure']
    df['Tenure_Ratio'] = df['Tenure'] / df['Age']
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Balance_Per_Product'] = df['Balance'] / df['NumOfProducts']
    df['Products_Per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['CreditScore_Age'] = df['CreditScore'] / df['Age']
    df['Has_Zero_Balance'] = (df['Balance'] == 0).astype(int)
    df['Engagement_Score'] = (
        df['IsActiveMember'] * 3 + df['NumOfProducts'] * 2 +
        (df['Balance'] > 50000).astype(int)
    )
    return df


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": metadata['model_name'],
        "f1_score": metadata['f1_optimal'],
        "threshold": threshold
    }


@app.post("/predict")
def predict(client: Client):
    try:
        df_client = pd.DataFrame([client.dict()])
        df_client = add_features(df_client)
        X_prep = preprocessor.transform(df_client)
        proba = float(model.predict_proba(X_prep)[0, 1])
        prediction = int(proba >= threshold)

        if proba > 0.8:
            risk_level = "CRITIQUE"
            action = "Contacter immediatement (offre premium)"
        elif proba > 0.6:
            risk_level = "ELEVE"
            action = "Contacter rapidement (offre personnalisee)"
        elif proba > 0.3:
            risk_level = "MOYEN"
            action = "Surveiller (email de fidelisation)"
        else:
            risk_level = "FAIBLE"
            action = "Aucune action necessaire"

        return {
            "churn_probability": round(proba, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "action_recommandee": action,
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)