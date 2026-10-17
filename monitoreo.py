"""Monitor de deriva de datos.

Compara, variable por variable, la distribución de los datos de entrenamiento con la
de los datos recientes de producción usando la prueba de Kolmogorov-Smirnov de dos
muestras. Si el p-valor es menor que el nivel de significancia, la variable derivó.
Termina con código 1 si hubo alguna alerta.

Uso: python monitoreo.py [ruta_produccion]
"""
import sys

import pandas as pd
from scipy.stats import ks_2samp

RUTA_ENTRENAMIENTO = "datos/clientes.csv"
RUTA_PRODUCCION = "datos/produccion.csv"
NIVEL_SIGNIFICANCIA = 0.01
VARIABLES = [
    "antiguedad_meses",
    "compras_ultimo_trimestre",
    "valor_promedio_compra",
    "dias_desde_ultima_compra",
    "quejas",
    "usa_app",
    "puntos_acumulados",
]


def detectar_deriva(entrenamiento: pd.DataFrame, produccion: pd.DataFrame) -> list[str]:
    alertas = []
    print(f"{'variable':<26}{'estadístico KS':>15}{'p-valor':>12}  resultado")
    for variable in VARIABLES:
        estadistico, p_valor = ks_2samp(entrenamiento[variable], produccion[variable])
        deriva = p_valor < NIVEL_SIGNIFICANCIA
        resultado = "ALERTA: deriva" if deriva else "OK"
        print(f"{variable:<26}{estadistico:>15.3f}{p_valor:>12.4g}  {resultado}")
        if deriva:
            alertas.append(variable)
    return alertas


if __name__ == "__main__":
    ruta_produccion = sys.argv[1] if len(sys.argv) > 1 else RUTA_PRODUCCION
    entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
    produccion = pd.read_csv(ruta_produccion)
    print(f"Entrenamiento: {len(entrenamiento)} filas | Producción: {len(produccion)} filas | "
          f"alfa = {NIVEL_SIGNIFICANCIA}\n")

    alertas = detectar_deriva(entrenamiento, produccion)
    if alertas:
        print(f"\nALERTA DE DERIVA en: {', '.join(alertas)}. Revisar la regla de reentrenamiento.")
        sys.exit(1)
    print("\nSin deriva: las distribuciones son compatibles con las de entrenamiento.")
