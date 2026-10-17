# Imagen liviana con la misma versión de Python usada para entrenar (3.13)
FROM python:3.13-slim

WORKDIR /app

# Primero las dependencias, para aprovechar la caché de Docker
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código del servicio y modelo aprobado por la puerta de calidad
COPY app.py .
COPY modelos/ modelos/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
