"""Pipeline de entrenamiento del modelo de abandono de clientes.

Versión 1: valida los datos y se detiene con un mensaje claro si algo está mal.
"""
import sys

import pandas as pd

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


if __name__ == "__main__":
    try:
        datos = validar_datos(RUTA_DATOS)
    except ValueError as error:
        print(f"VALIDACIÓN FALLIDA: {error}")
        sys.exit(1)
    print(f"Validación OK: {datos.shape[0]} filas, {datos.shape[1]} columnas, "
          f"{datos.isna().mean().max():.1%} de vacíos máximo por columna")
