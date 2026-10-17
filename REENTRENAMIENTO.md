# Regla de reentrenamiento

El modelo de riesgo de abandono se reentrena cuando se cumple **cualquiera** de estas condiciones.
Las revisa cada mes el responsable técnico del modelo (equipo de TI), con apoyo del analista de Mercadeo.

| # | Tipo | Condición concreta y medible | Cómo se mide |
|---|---|---|---|
| 1 | **Deriva de datos** | `monitoreo.py` alerta (KS, p < 0,01) sobre una misma variable **en 2 cortes mensuales seguidos**, o sobre **2 o más variables** en un solo corte. | `python monitoreo.py datos/produccion_<mes>.csv` sobre los clientes puntuados en el mes. |
| 2 | **Desempeño** | El **recall** medido con los abandonos reales del trimestre cae **por debajo de 0,50**, o la proporción de clientes marcados en riesgo cambia más de **±10 puntos** frente al mes de despliegue (hoy ~41 %). | Al cierre de cada trimestre se cruza la lista de clientes marcados con quienes realmente dejaron de comprar. |
| 3 | **Tiempo** | Han pasado **6 meses** desde el último entrenamiento, aunque no haya alertas. | Fecha del último entrenamiento registrada en MLflow. |

## Qué pasa al reentrenar

1. Se agregan al dataset los datos de los últimos meses ya etiquetados (se sabe si el cliente abandonó o no).
2. Se ejecuta `entrenar.py` con la misma configuración (y, si hace falta, nuevas variantes registradas en MLflow).
3. **Puerta de calidad:** el nuevo modelo solo reemplaza al actual si alcanza recall ≥ 0,50 en el conjunto de prueba
   y pasa las pruebas automáticas en GitHub Actions.
4. **Si el modelo reentrenado NO cumple el umbral:** no se despliega. El modelo actual sigue en servicio,
   pero la lista mensual se marca como "en revisión" y Mercadeo vuelve temporalmente a complementar la selección
   con su criterio manual. El equipo analiza la causa (cambio de comportamiento real, problema de datos, nuevas
   variables necesarias) en un plazo máximo de 2 semanas e informa a la Gerencia Comercial.

## Ejemplo con la alerta simulada

En la simulación de temporada baja, `dias_desde_ultima_compra` subió de 35 a 60 días en promedio
(KS = 0,47, p ≈ 1e-231). Con eso, el modelo pasa de marcar al **41 %** de los clientes como "en riesgo" al **58 %**:
la campaña de retención crecería ~44 % en costo sin evidencia de que el abandono real haya aumentado.
Esto activa la condición 2 (cambio > 10 puntos) y, si se repite el mes siguiente, la condición 1.
