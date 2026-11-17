"""
Entrena y guarda el modelo final del proyecto (Random Forest del Hito 2/3) para el MVP.

Replica exactamente la preparación del notebook `Deliverable 03_AML Project.ipynb`:
duplicados fuera, `contacto_previo` + `pdays` 999 -> 0, one-hot con drop_first,
sin `duration` (fuga), sin `emp.var.rate` ni `nr.employed` (VIF), escalado dentro del Pipeline,
mismos hiperparámetros (max_features=0.3, min_samples_leaf=20) y la misma partición 80/20 (seed 42),
para que el umbral de beneficio 0.296 del informe siga siendo válido.

Uso:  python train_model.py      ->  modelo/rf_bank.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = Path(__file__).parent
DATA = BASE / "bank-additional-full.csv"
SALIDA = BASE / "modelo" / "rf_bank.joblib"

CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERICAS = ["age", "campaign", "pdays", "previous", "cons.price.idx", "cons.conf.idx", "euribor3m"]
UMBRAL_BENEFICIO = 0.296   # elegido en el notebook sobre predicciones fuera de pliegue (5.3)


def preparar(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Misma transformación del notebook. Con `columnas`, alinea un registro nuevo a la matriz de entrenamiento."""
    df = df.copy()
    df["contacto_previo"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    df = df.drop(columns=[c for c in ["y", "duration", "emp.var.rate", "nr.employed"] if c in df])
    if columnas is None:
        return pd.get_dummies(df, columns=CATEGORICAS, drop_first=True, dtype=int)
    X = pd.get_dummies(df, columns=CATEGORICAS, dtype=int)   # sin drop_first: la base se descarta al alinear
    return X.reindex(columns=columnas, fill_value=0)


def main():
    df = pd.read_csv(DATA, sep=";").drop_duplicates().reset_index(drop=True)
    y = df["y"].map({"no": 0, "yes": 1})
    X = preparar(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    modelo = Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), NUMERICAS)], remainder="passthrough")),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                       max_features=0.3, min_samples_leaf=20,
                                       n_jobs=-1, random_state=SEED)),
    ]).fit(X_train, y_train)

    s = modelo.predict_proba(X_test)[:, 1]
    print(f"Registros: {len(df):,} · variables: {X.shape[1]}")
    print(f"AUC-PR prueba: {average_precision_score(y_test, s):.3f} · AUC-ROC: {roc_auc_score(y_test, s):.3f}")

    SALIDA.parent.mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo, "columnas": list(X.columns), "umbral": UMBRAL_BENEFICIO,
                 "tasa_base": float(y.mean())}, SALIDA, compress=3)
    print(f"Guardado en {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
