"""Simula datos de producción con deriva conocida en una sola variable.

Escenario: tres meses después del despliegue, una temporada baja hace que los
clientes espacien sus compras: dias_desde_ultima_compra aumenta en promedio 25 días.
Las demás variables se toman tal cual de una muestra de los datos originales.
Uso: python scripts/simular_produccion.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 7
VARIABLE_CON_DERIVA = "dias_desde_ultima_compra"
DESPLAZAMIENTO_DIAS = 25

if __name__ == "__main__":
    rng = np.random.default_rng(SEMILLA)
    entrenamiento = pd.read_csv("datos/clientes.csv")
    produccion = entrenamiento.sample(n=1500, random_state=SEMILLA).drop(columns="abandona")
    produccion[VARIABLE_CON_DERIVA] = np.clip(
        produccion[VARIABLE_CON_DERIVA] + DESPLAZAMIENTO_DIAS + rng.integers(-5, 6, len(produccion)),
        0, 730,
    )
    salida = Path("datos/produccion.csv")
    produccion.to_csv(salida, index=False)
    print(f"Guardado {salida}: {len(produccion)} filas. "
          f"'{VARIABLE_CON_DERIVA}' promedio: {entrenamiento[VARIABLE_CON_DERIVA].mean():.1f} -> "
          f"{produccion[VARIABLE_CON_DERIVA].mean():.1f} días")
