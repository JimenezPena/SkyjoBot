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


def _partida_4(c, rng, estado):
    pasos = 0
    while not estado["game_over"]:
        pasos += 1
        assert pasos < 6000
        if estado["turno_actual"] == 0:
            r = c.post('/api/4/step', json={"action": rng.choice([i for i, ok in enumerate(estado["action_mask"]) if ok])})
        else:
            r = c.post('/api/4/bot_step')
        assert r.status_code == 200, r.data
        estado = r.get_json(); _comprobar(estado)
    return estado


def test_niveles_de_4_jugadores_solo_los_validos_y_mezcla_de_bots():
    c = online_app.app.test_client()
    d = c.get('/api/4/niveles').get_json()
    assert {"heuristico", "aleatorio"} <= set(d["niveles"]) and "dqn_4.2" not in d["niveles"]
    assert d["por_defecto"] in d["niveles"]
    assert c.post('/api/4/reset', json={"niveles": ["dqn_4.2", "heuristico", "heuristico"]}).status_code == 400
    assert c.post('/api/4/reset', json={"niveles": ["heuristico"]}).status_code == 400
    r = c.post('/api/4/reset', json={"niveles": ["heuristico", "aleatorio", "heuristico"]})
    assert r.status_code == 200 and r.get_json()["niveles"] == ["heuristico", "aleatorio", "heuristico"]
    assert len(_partida_4(c, random.Random(5), r.get_json())["puntos"]) == 4


def test_un_modelo_v5_0_registrado_como_nivel_juega_en_la_web_de_4(tmp_path, monkeypatch):
    import json
    import torch
    import skyjo.niveles as niveles
    from skyjo.v5_0 import SkyjoDQN
    torch.manual_seed(0)
    torch.save(SkyjoDQN().state_dict(), tmp_path / "m.pth")
    ruta = tmp_path / "niveles.json"
    ruta.write_text(json.dumps({"heuristico": {"tipo": "heuristico"},
                                "dqn_5.0": {"tipo": "dqn", "version": "v5_0", "modelo": str(tmp_path / "m.pth")},
                                "dqn_4.2": {"tipo": "dqn", "version": "v4_2", "modelo": "modelos/v4_2.pth"}}))
    monkeypatch.setattr(niveles, "RUTA_NIVELES", str(ruta))
    monkeypatch.setattr(online_app, "_agentes", {})
    c = online_app.app.test_client()
    assert set(c.get('/api/4/niveles').get_json()["niveles"]) == {"heuristico", "dqn_5.0"}
    r = c.post('/api/4/reset', json={"niveles": ["dqn_5.0", "heuristico", "dqn_5.0"]})
    assert r.status_code == 200
    assert len(_partida_4(c, random.Random(6), r.get_json())["puntos"]) == 4


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


def test_el_nivel_elegido_en_el_desplegable_se_aplica_en_1v1():
    c = online_app.app.test_client()
    for nivel in ("aleatorio", "heuristico"):
        s = c.post('/api/reset', json={"niveles": [nivel]}).get_json()     # así lo envía index.html
        assert s["niveles"] == [nivel]
    assert c.post('/api/reset', json={"niveles": ["no_existe"]}).status_code == 400
    s = c.post('/api/step', json={"action": 1}).get_json()
    assert s["niveles"] == ["heuristico"]


def test_un_fallo_del_llm_no_bloquea_la_partida_web(monkeypatch):
    from skyjo.llm import AgenteLLM

    def cliente_roto(prompt, vista, razonar):
        raise TimeoutError("The read operation timed out")
    monkeypatch.setitem(online_app._agentes, "llm_falso", AgenteLLM(cliente_roto))
    c = online_app.app.test_client()
    s = c.post('/api/reset', json={"niveles": ["llm_falso"]}).get_json()
    for _ in range(40):
        if s["turno_actual"] == 0:
            acc = next(i for i, ok in enumerate(s["action_mask"]) if ok)
            s = c.post('/api/step', json={"action": acc}).get_json()
        else:
            r = c.post('/api/ia_step')
            assert r.status_code == 200, r.data               # antes: 500 y la partida se quedaba colgada
            s = r.get_json()
        if s["game_over"]:
            break
