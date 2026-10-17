"""Genera el dataset sintético (dummy) de clientes del programa de fidelización.

El riesgo de abandono depende de 4 columnas (quejas, dias_desde_ultima_compra,
antiguedad_meses y usa_app) más ruido, de modo que existe señal pero débil.
Uso: python scripts/generar_datos.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
N_FILAS = 5000
SALIDA = Path("datos/clientes.csv")


def generar(n: int = N_FILAS, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    antiguedad_meses = rng.integers(1, 121, n)
    compras_ultimo_trimestre = np.clip(rng.poisson(4, n), 0, 30)
    # Valor promedio en miles de pesos colombianos
    valor_promedio_compra = np.clip(rng.lognormal(4.4, 0.5, n), 10, 600).round(1)
    dias_desde_ultima_compra = np.clip(rng.exponential(35, n), 0, 365).astype(int)
    quejas = np.clip(rng.poisson(0.7, n), 0, 10)
    usa_app = rng.binomial(1, 0.45, n)
    puntos_acumulados = np.clip(
        antiguedad_meses * compras_ultimo_trimestre * rng.uniform(5, 15, n), 0, 50000
    ).astype(int)

    # Probabilidad de abandono: sube con quejas y días sin comprar,
    # baja con la antigüedad y el uso de la app. El ruido debilita la señal.
    ruido = rng.normal(0, 1.2, n)
    logit = (
        -1.35
        + 0.55 * quejas
        + 0.018 * dias_desde_ultima_compra
        - 0.014 * antiguedad_meses
        - 0.6 * usa_app
        + ruido
    )
    prob = 1 / (1 + np.exp(-logit))
    abandona = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "antiguedad_meses": antiguedad_meses,
            "compras_ultimo_trimestre": compras_ultimo_trimestre,
            "valor_promedio_compra": valor_promedio_compra,
            "dias_desde_ultima_compra": dias_desde_ultima_compra,
            "quejas": quejas,
            "usa_app": usa_app,
            "puntos_acumulados": puntos_acumulados,
            "abandona": abandona,
        }
    )


if __name__ == "__main__":
    df = generar()
    SALIDA.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"Guardado {SALIDA}: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Tasa de abandono: {df['abandona'].mean():.1%}")
