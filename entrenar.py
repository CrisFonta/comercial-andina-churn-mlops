"""Pipeline de entrenamiento del modelo de abandono de clientes.

Versión 3: valida los datos, entrena la variante pedida por línea de comandos,
registra el experimento en MLflow y aplica la puerta de calidad: el modelo solo
se guarda si el recall alcanza el umbral; si no, termina con código 1.

Ejemplo: python entrenar.py --modelo logistica --balancear --nombre logistica_balanceada
"""
import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEMILLA = 42
RUTA_DATOS = "datos/clientes.csv"
RUTA_MODELO = Path("modelos/modelo.joblib")
EXPERIMENTO = "abandono_clientes"
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
# Variante con menos variables: solo las de comportamiento reciente y servicio
COLUMNAS_REDUCIDAS = ["dias_desde_ultima_compra", "quejas", "usa_app", "antiguedad_meses"]
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


def construir_modelo(args) -> Pipeline:
    peso = "balanced" if args.balancear else None
    if args.modelo == "logistica":
        clasificador = LogisticRegression(
            C=args.c, class_weight=peso, max_iter=1000, random_state=SEMILLA
        )
    elif args.modelo == "bosque":
        clasificador = RandomForestClassifier(
            n_estimators=args.arboles, max_depth=args.profundidad,
            class_weight=peso, random_state=SEMILLA, n_jobs=-1,
        )
    else:  # gradiente: no tiene class_weight; el balanceo se hace con sample_weight al entrenar
        clasificador = GradientBoostingClassifier(
            n_estimators=args.arboles, max_depth=args.profundidad,
            learning_rate=0.05, random_state=SEMILLA,
        )
    # El escalado va dentro del pipeline para que el servicio aplique exactamente
    # la misma transformación que en el entrenamiento (evita training-serving skew).
    return Pipeline([("escalar", StandardScaler()), ("clasificador", clasificador)])


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Entrena el modelo de abandono de clientes")
    parser.add_argument("--modelo", choices=["logistica", "bosque", "gradiente"], default="logistica")
    parser.add_argument("--c", type=float, default=1.0, help="Regularización (logística)")
    parser.add_argument("--arboles", type=int, default=200, help="Número de árboles (bosque/gradiente)")
    parser.add_argument("--profundidad", type=int, default=5, help="Profundidad máxima (bosque/gradiente)")
    parser.add_argument("--balancear", action="store_true", help="Dar más peso a la clase minoritaria")
    parser.add_argument("--variables", choices=["todas", "reducidas"], default="todas")
    parser.add_argument("--corte", type=float, default=0.5,
                        help="Probabilidad a partir de la cual se marca 'en riesgo'")
    parser.add_argument("--umbral", type=float, default=0.50,
                        help="Recall mínimo para aprobar la puerta de calidad")
    parser.add_argument("--nombre", default=None, help="Nombre de la ejecución en MLflow")
    return parser.parse_args()


def main() -> int:
    args = leer_argumentos()

    try:
        datos = validar_datos(RUTA_DATOS)
    except ValueError as error:
        print(f"VALIDACIÓN FALLIDA: {error}")
        return 1
    print(f"Validación OK: {datos.shape[0]} filas, {datos.shape[1]} columnas")

    columnas = COLUMNAS_ENTRADA if args.variables == "todas" else COLUMNAS_REDUCIDAS
    X = datos[columnas]
    y = datos[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEMILLA, stratify=y
    )

    modelo = construir_modelo(args)
    if args.modelo == "gradiente" and args.balancear:
        pesos = compute_sample_weight("balanced", y_train)
        modelo.fit(X_train, y_train, clasificador__sample_weight=pesos)
    else:
        modelo.fit(X_train, y_train)

    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= args.corte).astype(int)
    metricas = calcular_metricas(y_test, pred, prob)
    aprobado = metricas["recall"] >= args.umbral

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=args.nombre):
        mlflow.log_params({
            "modelo": args.modelo,
            "c": args.c if args.modelo == "logistica" else None,
            "arboles": args.arboles if args.modelo != "logistica" else None,
            "profundidad": args.profundidad if args.modelo != "logistica" else None,
            "balancear": args.balancear,
            "variables": args.variables,
            "corte": args.corte,
            "umbral_recall": args.umbral,
        })
        mlflow.log_metrics(metricas)
        mlflow.set_tag("puerta_calidad", "aprobado" if aprobado else "rechazado")

        print(f"Modelo: {args.modelo} | balancear={args.balancear} | variables={args.variables} | corte={args.corte}")
        for nombre, valor in metricas.items():
            print(f"  {nombre:<10} {valor:.3f}")

        if aprobado:
            RUTA_MODELO.parent.mkdir(exist_ok=True)
            joblib.dump({"modelo": modelo, "columnas": columnas, "corte": args.corte}, RUTA_MODELO)
            mlflow.log_artifact(str(RUTA_MODELO))
            print(f"PUERTA DE CALIDAD APROBADA: recall {metricas['recall']:.3f} >= {args.umbral}. "
                  f"Modelo guardado en {RUTA_MODELO}")

    if not aprobado:
        print(f"PUERTA DE CALIDAD RECHAZADA: recall {metricas['recall']:.3f} < {args.umbral}. "
              "El modelo NO se guarda.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
