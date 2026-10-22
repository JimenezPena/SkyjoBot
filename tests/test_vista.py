import copy
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import torch

import skyjo.v4_2 as v4
from _referencia import obs_original, heuristico_original
from skyjo.agentes import AgenteHumano
from skyjo.consola import AgenteConsola, vista_a_texto
from skyjo.heuristico import AgenteAleatorio, AgenteHeuristico
from skyjo.niveles import cargar_nivel
from skyjo.partida import evaluar, jugar_partida
from skyjo.vista import construir_vista


def _recorrer(n, partidas, semilla=0):
    """Genera (env, jugador) en cada decisión de partidas jugadas con acciones legales al azar."""
    rng = random.Random(semilla)
    env = v4.SkyjoEnv(n)
    for _ in range(partidas):
        env.reset()
        done = False
        while not done:
            yield env
            a = rng.choice(np.flatnonzero(env.get_action_mask()).tolist())
            _, _, done, _ = env.step(a)


def test_la_vista_no_filtra_cartas_ocultas():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        for k, env in enumerate(_recorrer(n, 3)):
            if k % 7:
                continue
            otro = copy.deepcopy(env)
            ocultas = otro.tableros[:, :, :, 1] == 0
            otro.tableros[:, :, :, 0][ocultas] = rng.integers(-2, 13, size=int(ocultas.sum()))
            otro.mazo = list(rng.permutation(otro.mazo))
            v1, v2 = construir_vista(env), construir_vista(otro)
            assert np.array_equal(v1.valores, v2.valores) and np.array_equal(v1.estado, v2.estado)
            assert (v1.valores[v1.estado != 1] == 0).all()
            for ka, kb in zip(v4.observacion_desde_vista(v1).values(), v4.observacion_desde_vista(v2).values()):
                assert np.array_equal(ka, kb)


def test_observacion_4_2_desde_la_vista_es_identica_a_la_original():
    for n in (2, 3, 4):
        for env in _recorrer(n, 6, semilla=n):
            for j in {env.turno_actual, (env.turno_actual + 1) % n}:
                nueva, vieja = env.obtener_observacion(j), obs_original(env, j)
                assert np.array_equal(nueva['valores_tableros'], vieja['valores_tableros'])
                assert np.array_equal(nueva['visibilidad_tableros'], vieja['visibilidad_tableros'])
                np.testing.assert_allclose(nueva['escalares'], vieja['escalares'], rtol=0, atol=1e-6)


def test_heuristico_toma_las_mismas_decisiones_que_el_original():
    ag = AgenteHeuristico()
    for n in (2, 3, 4):
        for i, env in enumerate(_recorrer(n, 5, semilla=10 + n)):
            mask = env.get_action_mask()
            random.seed(i)
            vieja = heuristico_original(env, mask)
            random.seed(i)
            assert ag(construir_vista(env)) == vieja


def test_partida_de_4_asientos_con_agentes_mezclados():
    humano = AgenteHumano(lambda v: random.choice(v.acciones_legales()), nombre="humano")
    agentes = [AgenteHeuristico(), AgenteAleatorio(), humano, AgenteHeuristico()]
    r = jugar_partida(agentes)
    assert len(r.puntos) == 4 and not r.truncada
    res = evaluar(agentes, n_partidas=24, semilla=0)
    g = res.pop('_global')
    assert g['truncadas'] == 0 and set(res) == {"heuristico#1", "aleatorio", "humano", "heuristico#2"}
    assert all(v['partidas'] == 24 and v['ilegales'] == 0 for v in res.values())
    assert abs(sum(v['winrate'] * v['partidas'] for v in res.values()) - 24) < 1e-6      # un ganador (o reparto) por partida
    assert res["heuristico#1"]["winrate"] > res["aleatorio"]["winrate"]


def test_dqn_4_2_juega_desde_la_vista_y_solo_a_2_jugadores():
    dqn = cargar_nivel("dqn_4.2")
    res = evaluar([dqn, AgenteHeuristico()], n_partidas=6)
    assert res["_global"]["truncadas"] == 0 and res["dqn_4.2"]["ilegales"] == 0
    env = v4.SkyjoEnv(4)
    try:
        dqn(construir_vista(env))
        assert False, "debería rechazar 4 jugadores"
    except ValueError:
        pass


def test_consola_con_entradas_simuladas():
    env = v4.SkyjoEnv(3)
    vista = construir_vista(env)
    salidas, entradas = [], iter(["hola", "9 9 9", "1"])
    ag = AgenteConsola(entrada=lambda _: next(entradas), salida=salidas.append)
    assert ag(vista) == 1
    assert "Tú" in vista_a_texto(vista) and "Izquierda" in vista_a_texto(vista) and any("no válida" in s for s in salidas)
    assert AgenteConsola._parsear("p 2 3") == 2 + 1 * 4 + 2 and AgenteConsola._parsear("d 1 1") == 14


def test_evaluar_con_detalle_y_graficos(tmp_path):
    pytest = __import__('pytest')
    from skyjo.graficos import histogramas, imprimir_distribuciones
    ag = [AgenteHeuristico(), AgenteAleatorio(), AgenteHeuristico()]
    res = evaluar(ag, n_partidas=12, detalle=True)
    gs = res['_partidas']
    assert len(gs) == 12 and all(len(g['puntos']) == 3 for g in gs)
    assert '_partidas' in res and len([k for k in res if not k.startswith('_')]) == 3
    imprimir_distribuciones(gs)
    pytest.importorskip('matplotlib')
    rutas = histogramas(gs, str(tmp_path))
    assert len(rutas) == 5 and all(os.path.getsize(r) > 1000 for r in rutas)


def test_semilla_reproduce_el_reparto_con_partidas_intercaladas():
    from skyjo.v4_2 import SkyjoEnv
    a, b = SkyjoEnv(4, semilla=7), SkyjoEnv(4, semilla=8)
    a.reset(); b.reset()
    ref = SkyjoEnv(4, semilla=7); ref.reset()
    for _ in range(5):                          # consumir azar en otra partida no altera la primera
        b.reset()
    assert (a.tableros == ref.tableros).all() and a.mazo == ref.mazo
    r1 = jugar_partida([AgenteHeuristico()] * 3, semilla=3)
    r2 = jugar_partida([AgenteHeuristico()] * 3, semilla=3)
    assert r1.puntos == r2.puntos and r1.pasos == r2.pasos


def test_agente_llm_con_cliente_falso_y_razonamiento_opcional(tmp_path):
    from skyjo.llm import AgenteLLM, ClienteDQN, parsear_respuesta
    log = tmp_path / "llm.jsonl"
    ag = AgenteLLM(ClienteDQN(AgenteHeuristico()), razonar=False, registro=str(log))
    r = jugar_partida([ag, AgenteHeuristico()], semilla=1)
    assert ag.fallos == 0 and ag.ultimo_motivo is None and r.ilegales[0] == 0
    prompts = [__import__('json').loads(l) for l in log.read_text().splitlines()]
    assert len(prompts) == ag.llamadas and '"motivo"' not in prompts[0]['prompt'] and prompts[0]['motivo'] is None
    ag.razonar = True
    jugar_partida([ag, AgenteHeuristico()], semilla=1, max_pasos=3)
    assert ag.ultimo_motivo == "hago esta jugada porque sí"
    # una respuesta basura se reintenta y luego juega el heurístico
    malo = AgenteLLM(lambda p, v, r: "no sé", reintentos=1)
    jugar_partida([malo, AgenteHeuristico()], semilla=1, max_pasos=4)
    assert malo.fallos >= 1 and malo.llamadas == 2 * malo.fallos
    assert parsear_respuesta('texto {"accion": 99}', None.__class__ and type('V', (), {'mask': [True] * 26})) == (None, None)


def test_nivel_llm_falso_oculto_y_cargable():
    from skyjo.niveles import listar_niveles, niveles_para
    # el nivel de Gemini gasta cuota y clave: no debe salir en la web (llm_falso sí puede, es gratis)
    assert 'llm_gemini' not in listar_niveles() and 'llm_gemini' not in niveles_para(2)
    assert 'llm_gemini' in listar_niveles(ocultos=True)
    ag = cargar_nivel('llm_falso')
    r = jugar_partida([ag, AgenteHeuristico()], semilla=2)
    assert ag.base.fallos == 0 if hasattr(ag, 'base') else ag.fallos == 0


def test_match_completo_hasta_100_puntos():
    from skyjo.match import evaluar_matches, jugar_match
    ag = [AgenteHeuristico(), AgenteHeuristico()]
    m = jugar_match(ag)
    assert max(m['total']) >= 100 and m['rondas'] >= 2 and m['ganadores']
    assert all(m['total'][g] == min(m['total']) for g in m['ganadores'])
    res = evaluar_matches(ag, n_matches=4, semilla=0)
    assert abs(sum(v['winrate'] for k, v in res.items() if not k.startswith('_')) - 1) < 1e-9


def test_cliente_gemini_sin_red():
    import json
    import pytest
    from skyjo.llm import AgenteLLM, ClienteGemini, CuotaAgotada
    esperas, respuestas = [], iter([
        (429, {"error": {"details": [{"retryDelay": "7s"}]}}),
        (503, {}),
        (200, {"candidates": [{"content": {"parts": [{"text": '{"accion": 1}'}]}}],
               "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": 3}})])
    cuerpos = []

    def post(url, cuerpo):
        cuerpos.append(cuerpo)
        return next(respuestas)
    c = ClienteGemini(_post=post, _dormir=esperas.append, rpm=0)
    assert json.loads(c("hola")) == {"accion": 1}
    assert esperas == [7.0, 4.0] and (c.tokens_entrada, c.tokens_salida, c.llamadas) == (100, 8, 3)
    g = cuerpos[0]["generationConfig"]
    assert g["responseMimeType"] == "application/json" and g["thinkingConfig"]["thinkingBudget"] == 0
    # cuota diaria: parar, no reintentar
    d = ClienteGemini(_post=lambda u, b: (429, {"error": {"message": "RequestsPerDay"}}), _dormir=esperas.append, rpm=0)
    with pytest.raises(CuotaAgotada):
        d("hola")
    # ritmo: dos llamadas seguidas esperan 60/rpm
    t, dormidas = [0.0], []
    r = ClienteGemini(_post=lambda u, b: (200, {}), rpm=10, _dormir=dormidas.append, _reloj=lambda: t[0])
    r("a"); r("b")
    assert dormidas == [6.0]
    # el agente tolera respuestas vacías (sin 'candidates')
    ag = AgenteLLM(r, reintentos=0)
    jugar_partida([ag, AgenteHeuristico()], semilla=1, max_pasos=3)
    assert ag.fallos >= 1


def test_prompt_con_ayudas_motivo_primero_y_estrategia():
    from skyjo.llm import construir_prompt, ESTRATEGIA
    from skyjo.v4_2 import SkyjoEnv
    env = SkyjoEnv(2, semilla=3); env.reset()
    env.tableros[0, :, 3, 0] = [6, 6, 9]; env.tableros[0, :, 3, 1] = [1, 1, 0]
    env.fase_turno, env.carta_en_mano, env.origen_robo = 2, 6, 'MAZO'
    v = construir_vista(env, 0)
    base = construir_prompt(v, razonar=False)
    assert 'completa la columna' not in base and ESTRATEGIA not in base and '"motivo"' not in base
    p = construir_prompt(v, razonar=True, ayudas=True, estrategia=True)
    assert '13: coloca la 6 en fila 3, col 4 → completa la columna 4 de 6s y la elimina: -17 puntos' in p
    assert ESTRATEGIA in p and p.rstrip().endswith('{"motivo": "<razonamiento breve>", "accion": <número>}')
    assert construir_prompt(v, razonar=True, motivo_primero=False).rstrip().endswith('{"accion": <número>, "motivo": "<una frase>"}')


def test_evaluar_con_semilla_es_reproducible_y_no_toca_el_azar_global():
    ag = [AgenteHeuristico(), AgenteHeuristico(), AgenteAleatorio()]
    random.seed(99)
    antes = random.getstate()
    r1 = evaluar(ag, n_partidas=6, semilla=5)
    assert random.getstate() == antes                 # el azar de quien llama (p. ej. el entrenamiento) no cambia
    r2 = evaluar(ag, n_partidas=6, semilla=5)
    k = [x for x in r1 if not x.startswith('_')]
    assert [r1[x]['puntos'] for x in k] == [r2[x]['puntos'] for x in k]
    r3 = evaluar(ag, n_partidas=6, semilla=6)
    assert [r1[x]['puntos'] for x in k] != [r3[x]['puntos'] for x in k]


def test_gemini_reintenta_timeouts_y_no_cuelga_la_partida(monkeypatch):
    import urllib.request
    import pytest
    from skyjo.llm import ClienteGemini
    # 1) un timeout de red se trata como error transitorio y se reintenta
    def lanza(*a, **k):
        raise TimeoutError("The read operation timed out")
    monkeypatch.setattr(urllib.request, "urlopen", lanza)
    esperas = []
    c = ClienteGemini(api_key="x", rpm=0, reintentos_http=2, _dormir=esperas.append)
    with pytest.raises(RuntimeError, match="599"):
        c("hola")
    assert c.llamadas == 3 and len(esperas) == 2
    # 2) si el timeout es pasajero, la siguiente llamada funciona
    seq = iter([(599, {}), (200, {"candidates": [{"content": {"parts": [{"text": '{"accion": 1}'}]}}]})])
    c2 = ClienteGemini(_post=lambda u, b: next(seq), rpm=0, _dormir=lambda s: None)
    assert '"accion": 1' in c2("hola")


def test_gemini_se_adapta_a_modelos_que_rechazan_el_pensamiento_y_a_503_largos(monkeypatch):
    import pytest
    import json
    from skyjo.llm import ClienteGemini, crear_agente_llm
    cuerpos = []
    bien = (200, {"candidates": [{"content": {"parts": [{"text": '{"accion": 1}'}]}}]})

    def post(url, cuerpo):
        cuerpos.append(cuerpo["generationConfig"])
        return (400, {"error": {"message": "Request contains an invalid argument."}}) if "thinkingConfig" in cuerpo["generationConfig"] else bien
    c = ClienteGemini(_post=post, rpm=0, _dormir=lambda s: None)
    assert json.loads(c("a")) == {"accion": 1}
    assert "thinkingConfig" in cuerpos[0] and "thinkingConfig" not in cuerpos[1]
    c("b")
    assert "thinkingConfig" not in cuerpos[-1] and len(cuerpos) == 3          # recuerda que no debe enviarlo
    # niveles de pensamiento (Gemini 3) y "sin ajuste"
    g = ClienteGemini(_post=lambda u, b: (cuerpos.append(b["generationConfig"]) or bien), rpm=0, nivel_pensamiento="low")
    g("x"); assert cuerpos[-1]["thinkingConfig"] == {"thinkingLevel": "low"}
    n = ClienteGemini(_post=lambda u, b: (cuerpos.append(b["generationConfig"]) or bien), rpm=0, pensar=None)
    n("x"); assert "thinkingConfig" not in cuerpos[-1]
    # 503 sostenidos: la espera crece hasta espera_max y se admiten más reintentos
    esperas = []
    k = ClienteGemini(_post=lambda u, b: (503, {}), rpm=0, reintentos_http=6, espera_max=10, _dormir=esperas.append)
    with pytest.raises(RuntimeError):
        k("x")
    assert esperas == [2.0, 4.0, 8.0, 10.0, 10.0, 10.0]
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    ag = crear_agente_llm(dict(cliente="gemini", modelo="m", pensar=None, reintentos_http=8, espera_max=120,
                               nivel_pensamiento="minimal"))
    assert ag.cliente.reintentos_http == 8 and ag.cliente.espera_max == 120 and ag.cliente.pensar is None
