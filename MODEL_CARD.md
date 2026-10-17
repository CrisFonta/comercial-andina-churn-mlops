# Model card · Riesgo de abandono de clientes (Comercial Andina)

## Qué hace
Estima la probabilidad de que un cliente del programa de fidelización deje de comprar. Si la probabilidad es
≥ 0,50, el cliente se marca "en riesgo" y se sugiere incluirlo en la campaña mensual de retención.
Se consulta como servicio web (`POST /predecir`).

## Modelo
- Regresión logística con escalado estándar y pesos balanceados por clase (`class_weight="balanced"`).
- 4 variables de entrada: `dias_desde_ultima_compra`, `quejas`, `usa_app`, `antiguedad_meses`.
- Efecto de cada variable (coeficientes estandarizados): más días sin comprar (+0,51) y más quejas (+0,42)
  aumentan el riesgo; mayor antigüedad (−0,41) y usar la app (−0,26) lo reducen.
- Elegido entre 7 experimentos registrados en MLflow (ver `DEFINICION.md` y el informe).

## Datos de entrenamiento
- **Datos sintéticos (dummy)**, 5 000 clientes, generados con `scripts/generar_datos.py` (semilla 42).
  No corresponden a personas reales. Tasa de abandono: 26,9 %.
- División 75 % entrenamiento / 25 % prueba, estratificada, semilla 42.

## Desempeño (conjunto de prueba, 1 250 clientes)
| Métrica | Valor |
|---|---|
| **Recall (criterio de éxito, umbral 0,50)** | **0,598** |
| Precisión | 0,396 |
| Exactitud | 0,646 |
| F1 | 0,476 |
| ROC AUC | 0,710 |

De cada 10 clientes que realmente se van, el modelo detecta ~6. De cada 10 clientes que marca en riesgo,
~4 realmente se iban a ir: el resto recibe una acción de retención que no necesitaba (costo aceptado, ver `DEFINICION.md`).

## Usos previstos
- Priorizar a qué clientes dirigir la campaña mensual de retención (llamada, cupón, puntos dobles).
- Apoyar, no reemplazar, el criterio del equipo de Mercadeo.

## Usos no recomendados
- Tomar decisiones que perjudiquen al cliente (negar beneficios, subir precios, excluirlo de promociones).
- Decisiones de crédito, cobranza o empleo.
- Usarlo en otras cadenas, otras regiones o con clientes fuera del programa de fidelización sin reentrenarlo.
- Tratar la probabilidad como certeza sobre un cliente individual.

## Limitaciones
- Entrenado con datos sintéticos de señal débil: las cifras sirven para demostrar el proceso, no para
  estimar el desempeño real. Con datos reales debe reentrenarse y validarse de nuevo.
- La precisión es baja (0,40): ~6 de cada 10 contactos son "falsas alarmas".
- No considera estacionalidad: una temporada baja eleva los días sin comprar y dispara falsas alarmas
  (ver `REENTRENAMIENTO.md`).
- No se ha evaluado el sesgo por tienda, ciudad, edad o canal, porque los datos no incluyen esas variables.
  Debe hacerse antes de un uso real.

## Mantenimiento
- Responsable técnico: equipo de TI de Comercial Andina (modelo, servicio, monitoreo).
- Responsable de negocio: Gerencia Comercial / Mercadeo (umbral, uso de la lista, resultados de la campaña).
- Monitoreo mensual con `monitoreo.py`; reentrenamiento según `REENTRENAMIENTO.md`.
- Versión 1.0 · octubre de 2026.
