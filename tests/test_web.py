"""La web (online_app + templates/index.html) debe funcionar siempre. Ejecutar: python -m pytest -q tests"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import online_app
from skyjo.niveles import listar_niveles

CLAVES = {"tableros", "turno_actual", "fase_turno", "origen_robo", "carta_en_mano",
          "top_descarte", "game_over", "puntos", "action_mask"}


def _comprobar(s):
    assert CLAVES <= set(s), set(s)
    for jugador in s["tableros"]:                 # ninguna carta oculta debe filtrarse
        for fila in jugador:
            for valor, vis in fila:
                if vis != 1:
                    assert valor == 0


def _partida(nivel):
    c = online_app.app.test_client()
    r = c.post('/api/reset', json={"nivel": nivel} if nivel else None)   # sin cuerpo = como hace index.html
    assert r.status_code == 200, r.data
    s = r.get_json(); _comprobar(s)
    rng, pasos = random.Random(0), 0
    while not s["game_over"]:
        pasos += 1
        assert pasos < 3000
        if s["turno_actual"] == 0:
            acc = rng.choice([i for i, ok in enumerate(s["action_mask"]) if ok])
            r = c.post('/api/step', json={"action": acc})
        else:
            r = c.post('/api/ia_step')
        assert r.status_code == 200, (nivel, r.data)
        s = r.get_json(); _comprobar(s)
    assert len(s["puntos"]) == 2


def test_partida_completa_con_cada_nivel():
    for nivel in listar_niveles():
        _partida(nivel)


def test_partida_con_nivel_por_defecto():
    _partida(None)


def test_nivel_desconocido_y_listado():
    c = online_app.app.test_client()
    assert c.post('/api/reset', json={"nivel": "no_existe"}).status_code == 400
    d = c.get('/api/niveles').get_json()
    assert online_app.NIVEL_POR_DEFECTO in d["niveles"]


def test_partida_4_jugadores_humano_contra_3_bots():
    c = online_app.app.test_client()
    assert c.get('/cuatro').status_code == 200
    s = c.post('/api/4/reset').get_json(); _comprobar(s)
    assert len(s["tableros"]) == 4
    rng, pasos = random.Random(1), 0
    while not s["game_over"]:
        pasos += 1
        assert pasos < 6000
        if s["turno_actual"] == 0:
            acc = rng.choice([i for i, ok in enumerate(s["action_mask"]) if ok])
            r = c.post('/api/4/step', json={"action": acc})
        else:
            assert c.post('/api/4/step', json={"action": 0}).status_code == 400   # no es tu turno
            r = c.post('/api/4/bot_step')
        assert r.status_code == 200, r.data
        s = r.get_json(); _comprobar(s)
    assert len(s["puntos"]) == 4


def test_index_solo_usa_rutas_que_existen():
    c = online_app.app.test_client()
    r = c.get('/'); assert r.status_code == 200
    assert b'href="/cuatro"' in r.data and b'href="/"' in c.get('/cuatro').data   # enlaces entre los dos modos
    usadas = set()
    for plantilla in ('index.html', 'cuatro.html'):
        html = open(os.path.join(os.path.dirname(__file__), '..', 'templates', plantilla), encoding='utf-8').read()
        usadas |= set(re.findall(r"fetch\(['\"](/api/[a-z_]+)", html))
        usadas |= {"/api/4" + r for r in re.findall(r"fetch\(API \+ '(/[a-z_]+)", html)}
    assert usadas, "index.html no llama a la API?"
    rutas = {r.rule for r in online_app.app.url_map.iter_rules()}
    assert usadas <= rutas, usadas - rutas


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
