"""Servicio web que expone el modelo de riesgo de abandono.

Arranque: uvicorn app:app --port 8000   (documentación en http://localhost:8000/docs)
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

RUTA_MODELO = Path("modelos/modelo.joblib")

# Se carga una sola vez, al arrancar. Incluye el pipeline (escalado + modelo),
# las columnas que espera y el corte de probabilidad usado en el entrenamiento.
artefacto = joblib.load(RUTA_MODELO)
modelo = artefacto["modelo"]
columnas = artefacto["columnas"]
corte = artefacto["corte"]

app = FastAPI(
    title="Comercial Andina · Riesgo de abandono",
    description="Estima la probabilidad de que un cliente del programa de fidelización deje de comprar.",
    version="1.0.0",
)


class Cliente(BaseModel):
    """Datos de un cliente. Los límites son las mismas reglas de DEFINICION.md."""
    antiguedad_meses: int = Field(..., ge=0, le=240, description="Meses en el programa")
    dias_desde_ultima_compra: int = Field(..., ge=0, le=730, description="Días desde la última compra")
    quejas: int = Field(..., ge=0, le=20, description="Quejas en el último año")
    usa_app: int = Field(..., ge=0, le=1, description="1 si usa la app, 0 si no")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"antiguedad_meses": 8, "dias_desde_ultima_compra": 95, "quejas": 3, "usa_app": 0}
            ]
        }
    }


@app.get("/salud")
def salud():
    return {"estado": "activo", "modelo_cargado": True}


@app.post("/predecir")
def predecir(cliente: Cliente):
    # Mismo orden de columnas que en el entrenamiento; el pipeline aplica el mismo escalado.
    fila = pd.DataFrame([cliente.model_dump()])[columnas]
    probabilidad = float(modelo.predict_proba(fila)[0, 1])
    en_riesgo = probabilidad >= corte
    return {
        "probabilidad_abandono": round(probabilidad, 4),
        "en_riesgo": bool(en_riesgo),
        "accion_sugerida": "Incluir en campaña de retención" if en_riesgo else "Sin acción",
    }
