# Comercial Andina · Riesgo de abandono de clientes (MLOps)

Proyecto del Taller integrador de la Unidad 3 (Estrategias de Transformación Digital, Universidad de Caldas).
Línea **A. Clientes**: predecir qué clientes del programa de fidelización están en riesgo de dejar de comprar (`abandona`).

> Los datos son **sintéticos (dummy)**. Comercial Andina es una empresa ficticia.

## Archivos

| Archivo | Qué hace |
|---|---|
| `scripts/generar_datos.py` | Genera `datos/clientes.csv` (5 000 filas, semilla 42) |
| `DEFINICION.md` | Problema, columnas, reglas de validación y criterio de éxito (recall ≥ 0,50) |
| `entrenar.py` | Valida datos, entrena, registra en MLflow y aplica la puerta de calidad |
| `app.py` | Servicio FastAPI: `GET /salud`, `POST /predecir` |
| `pruebas/test_app.py` | 6 pruebas automáticas del servicio |
| `Dockerfile`, `requirements-api.txt` | Empaquetado del servicio |
| `.github/workflows/pruebas.yml` | CI: en cada push entrena desde cero y ejecuta las pruebas |
| `scripts/simular_produccion.py` | Crea `datos/produccion.csv` con deriva en `dias_desde_ultima_compra` |
| `monitoreo.py` | Detecta deriva con Kolmogorov-Smirnov (alfa = 0,01) |
| `REENTRENAMIENTO.md` | Cuándo y cómo se reentrena |
| `MODEL_CARD.md` | Ficha del modelo |

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python entrenar.py --modelo logistica --balancear --variables reducidas   # modelo elegido
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1             # comparar experimentos
uvicorn app:app --port 8000                                               # servicio (http://localhost:8000/docs)
python -m pytest -v pruebas                                               # pruebas
docker build -t comercial-andina-abandono . && docker run -p 8000:8000 comercial-andina-abandono
python monitoreo.py                                                       # monitor de deriva
```

## El ciclo de MLOps en este proyecto

| Etapa | Pieza que la cubre | Qué pasaría sin ella |
|---|---|---|
| Definición del problema | `DEFINICION.md` | No habría criterio para aprobar o rechazar un modelo; se habría elegido el de mayor exactitud (0,76) que solo detecta 2 de cada 10 clientes que se van. |
| Datos | `generar_datos.py` + validación en `entrenar.py` | Un archivo con días negativos o columnas faltantes entrenaría un modelo defectuoso sin que nadie lo note. |
| Experimentación y entrenamiento | `entrenar.py` + MLflow | Las 7 variantes serían archivos sueltos y nadie sabría qué configuración dio recall 0,598. |
| Validación | Puerta de calidad (recall ≥ 0,50) | El modelo base, con recall de 0,20, habría llegado al servicio. |
| Empaquetado | `Dockerfile` | El servicio podría fallar en el servidor por otra versión de Python o scikit-learn. |
| Despliegue | `app.py` (FastAPI) + GitHub Actions | El modelo sería un archivo que Mercadeo no puede consultar; un cambio de código podría romperlo sin aviso. |
| Monitoreo | `monitoreo.py` + `REENTRENAMIENTO.md` | En temporada baja el modelo marcaría al 58 % de los clientes (vs 41 %) y la campaña se encarecería sin que nadie supiera por qué. |
