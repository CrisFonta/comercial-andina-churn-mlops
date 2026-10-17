"""Pruebas automáticas del servicio de riesgo de abandono."""
from fastapi.testclient import TestClient

from app import app

cliente = TestClient(app)

CLIENTE_VALIDO = {
    "antiguedad_meses": 8,
    "dias_desde_ultima_compra": 95,
    "quejas": 3,
    "usa_app": 0,
}


def test_salud_responde_200():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "activo"


def test_prediccion_valida_tiene_formato_esperado():
    respuesta = cliente.post("/predecir", json=CLIENTE_VALIDO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"probabilidad_abandono", "en_riesgo", "accion_sugerida"}
    assert 0.0 <= cuerpo["probabilidad_abandono"] <= 1.0
    assert isinstance(cuerpo["en_riesgo"], bool)


def test_valor_negativo_se_rechaza():
    datos = {**CLIENTE_VALIDO, "quejas": -2}
    assert cliente.post("/predecir", json=datos).status_code == 422


def test_usa_app_fuera_de_rango_se_rechaza():
    datos = {**CLIENTE_VALIDO, "usa_app": 5}
    assert cliente.post("/predecir", json=datos).status_code == 422


def test_campo_faltante_se_rechaza():
    datos = {k: v for k, v in CLIENTE_VALIDO.items() if k != "quejas"}
    assert cliente.post("/predecir", json=datos).status_code == 422


def test_cliente_de_alto_riesgo_tiene_mas_probabilidad_que_uno_fiel():
    fiel = {"antiguedad_meses": 96, "dias_desde_ultima_compra": 5, "quejas": 0, "usa_app": 1}
    p_riesgo = cliente.post("/predecir", json=CLIENTE_VALIDO).json()["probabilidad_abandono"]
    p_fiel = cliente.post("/predecir", json=fiel).json()["probabilidad_abandono"]
    assert p_riesgo > p_fiel
