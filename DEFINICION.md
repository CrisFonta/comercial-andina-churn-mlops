# Definición del problema

## Qué predice el modelo y qué decisión apoya

El modelo estima la probabilidad de que un cliente del programa de fidelización de Comercial Andina
**deje de comprar** (`abandona = 1`) en el próximo periodo. Con esa probabilidad, el área de Mercadeo
arma cada mes la **lista de clientes en riesgo** a los que se dirige la campaña de retención
(llamada del call center, cupón o puntos dobles). Hoy esa lista se arma a mano y por experiencia.

## Datos

- Archivo: `datos/clientes.csv`, **sintético (dummy)**, generado con `scripts/generar_datos.py`
  (semilla fija 42, reproducible).
- **5 000 filas** y 8 columnas (7 de entrada + objetivo). Sin valores vacíos.
- La probabilidad de abandono se generó a partir de 4 columnas (quejas, días desde la última compra,
  antigüedad y uso de la app) más ruido normal, por lo que hay señal pero débil.
- Tasa de abandono: **26,9 %** (clases desbalanceadas).

## Columnas y rangos razonables (reglas de validación)

| Columna | Tipo | Significado | Rango válido |
|---|---|---|---|
| antiguedad_meses | entero | Meses en el programa | 0 – 240 |
| compras_ultimo_trimestre | entero | Número de compras en los últimos 3 meses | 0 – 60 |
| valor_promedio_compra | decimal | Ticket promedio, en miles de pesos | > 0 y ≤ 2 000 |
| dias_desde_ultima_compra | entero | Días desde la última compra | 0 – 730 |
| quejas | entero | Quejas registradas en el último año | 0 – 20 |
| usa_app | 0/1 | Usa la app de la tienda | 0 o 1 |
| puntos_acumulados | entero | Puntos de fidelización disponibles | 0 – 100 000 |
| **abandona** (objetivo) | 0/1 | Dejó de comprar en el periodo | 0 o 1 |

Reglas adicionales: deben estar todas las columnas, al menos 1 500 filas y no más de 5 % de vacíos por columna.

## Criterio de éxito

- **Métrica: recall de la clase `abandona = 1`** (de los clientes que realmente se van, qué proporción detecta el modelo).
- **Umbral mínimo: recall ≥ 0,50** sobre el conjunto de prueba (25 %, estratificado, semilla 42).

**Por qué recall y no exactitud.** Solo el 27 % de los clientes abandona: un modelo que dijera "nadie se va"
tendría 73 % de exactitud y no serviría para nada. Para Comercial Andina los dos errores no cuestan lo mismo:

- **Falso negativo** (no detectar a quien se va): se pierde el margen futuro del cliente,
  estimado en ~$480 000 al año (ticket promedio de ~$92 000 × ~16 compras al año × ~33 % de margen bruto).
- **Falso positivo** (contactar a quien no se iba a ir): cuesta la acción de retención, estimada en ~$25 000
  (llamada + cupón).

Como perder un cliente cuesta mucho más que contactar a uno de más, se prioriza detectar
a los que se van (recall), vigilando que la precisión no caiga tanto que la campaña se vuelva cara.

**Por qué el umbral es bajo.** Los datos son sintéticos y la señal es débil a propósito. En un proyecto real
el umbral lo acordaría el negocio (Gerencia Comercial y Mercadeo) con base en el presupuesto de la campaña.
