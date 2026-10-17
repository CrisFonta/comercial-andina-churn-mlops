"""Pipeline de entrenamiento del modelo de abandono de clientes.

Versión 2: valida los datos y entrena un modelo base con métricas sobre prueba.
"""
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
RUTA_DATOS = "datos/clientes.csv"
OBJETIVO = "abandona"
COLUMNAS_ENTRADA = [
    "antiguedad_meses",
    "compras_ultimo_trimestre",
    "valor_promedio_compra",
    "dias_desde_ultima_compra",
    "quejas",
    "usa_app",
    "puntos_acumulados",
]
MIN_FILAS = 1500
MAX_VACIOS = 0.05
# Rangos razonables tomados de DEFINICION.md
RANGOS = {
    "antiguedad_meses": (0, 240),
    "compras_ultimo_trimestre": (0, 60),
    "valor_promedio_compra": (0.01, 2000),
    "dias_desde_ultima_compra": (0, 730),
    "quejas": (0, 20),
    "usa_app": (0, 1),
    "puntos_acumulados": (0, 100000),
}


def validar_datos(ruta: str) -> pd.DataFrame:
    """Lee los datos y lanza ValueError si no cumplen las reglas de DEFINICION.md."""
    df = pd.read_csv(ruta)

    faltantes = set(COLUMNAS_ENTRADA + [OBJETIVO]) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    if len(df) < MIN_FILAS:
        raise ValueError(f"Hay {len(df)} filas; se requieren al menos {MIN_FILAS}")

    vacios = df.isna().mean()
    excedidas = vacios[vacios > MAX_VACIOS]
    if not excedidas.empty:
        detalle = ", ".join(f"{c}={v:.1%}" for c, v in excedidas.items())
        raise ValueError(f"Columnas con más de {MAX_VACIOS:.0%} de vacíos: {detalle}")

    valores_objetivo = set(df[OBJETIVO].dropna().unique())
    if not valores_objetivo <= {0, 1}:
        raise ValueError(f"'{OBJETIVO}' solo puede ser 0 o 1; se encontró {sorted(valores_objetivo)}")

    if not set(df["usa_app"].dropna().unique()) <= {0, 1}:
        raise ValueError("'usa_app' solo puede ser 0 o 1")

    for col, (minimo, maximo) in RANGOS.items():
        fuera = df[(df[col] < minimo) | (df[col] > maximo)]
        if not fuera.empty:
            raise ValueError(
                f"'{col}' tiene {len(fuera)} valores fuera del rango [{minimo}, {maximo}] "
                f"(ej. fila {fuera.index[0]}: {fuera[col].iloc[0]})"
            )

    return df


def calcular_metricas(y_real, y_pred, y_prob) -> dict:
    return {
        "exactitud": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred),
        "f1": f1_score(y_real, y_pred),
        "roc_auc": roc_auc_score(y_real, y_prob),
    }


if __name__ == "__main__":
    try:
        datos = validar_datos(RUTA_DATOS)
    except ValueError as error:
        print(f"VALIDACIÓN FALLIDA: {error}")
        sys.exit(1)
    print(f"Validación OK: {datos.shape[0]} filas, {datos.shape[1]} columnas, "
          f"{datos.isna().mean().max():.1%} de vacíos máximo por columna")

    X = datos[COLUMNAS_ENTRADA]
    y = datos[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEMILLA, stratify=y
    )

    # Modelo base: regresión logística. El escalado va dentro del pipeline para que
    # el servicio aplique exactamente la misma transformación (evita training-serving skew).
    modelo = Pipeline([
        ("escalar", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
    ])
    modelo.fit(X_train, y_train)

    metricas = calcular_metricas(
        y_test, modelo.predict(X_test), modelo.predict_proba(X_test)[:, 1]
    )
    for nombre, valor in metricas.items():
        print(f"  {nombre:<10} {valor:.3f}")
