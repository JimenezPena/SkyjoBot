"""Agente que decide con un LLM. La interfaz con el modelo es un "cliente" intercambiable:

    cliente(prompt: str, vista: Vista, razonar: bool) -> str      # respuesta JSON: {"accion": 7, "motivo": "..."}

  - ClienteDQN (en niveles.json: "cliente": "falso"): no llama a ningún LLM; juega con un nivel ya entrenado
    (p. ej. dqn_5.1) y responde con el mismo formato JSON. Sirve para probar todo el circuito sin gastar cuota.
  - ClienteGemini ("cliente": "gemini"): API de Google (por defecto gemini-2.5-flash), clave en la variable de entorno
    GEMINI_API_KEY. Controla el ritmo de llamadas (cuota gratuita), reintenta los errores transitorios y cuenta tokens.
  - Cualquier otro cliente solo tiene que implementar esa misma función; el agente no cambia.

El razonamiento se activa con razonar=True (agente, "razonar": true en niveles.json o --razonar en la consola):
añade al prompt la petición de un "motivo" breve, que se guarda en agente.ultimo_motivo y en el registro.
Con razonar=False el prompt pide solo {"accion": n} y se ahorran tokens de salida.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from .agentes import Agente
from .consola import describir_accion, vista_a_texto
from .heuristico import AgenteHeuristico

REGLAS = (
    "Juegas a Skyjo (menos puntos es mejor). Cada jugador tiene 12 cartas en 3 filas x 4 columnas; '?' = oculta, "
    "'--' = columna eliminada (3 iguales). En tu turno robas del mazo o coges el descarte y: colocas la carta en una "
    "casilla (la anterior va al descarte) o, si robaste del mazo, la descartas y destapas una carta oculta. "
    "Quien destapa todo cierra la ronda; si no tiene la puntuación más baja, se le dobla."
)


def construir_prompt(vista, razonar=False):
    acciones = "\n".join(f"{a}: {describir_accion(a, vista)}" for a in vista.acciones_legales())
    formato = ('{"accion": <número>, "motivo": "<una frase>"}' if razonar else '{"accion": <número>}')
    return (f"{REGLAS}\n\n{vista_a_texto(vista)}\n\nAcciones legales:\n{acciones}\n\n"
            f"Responde solo con JSON: {formato}")


def parsear_respuesta(texto, vista):
    """(accion, motivo) o (None, None) si la respuesta no es JSON válido o la acción no es legal."""
    m = re.search(r"\{.*\}", texto or "", re.S)
    try:
        d = json.loads(m.group(0))
        a = int(d["accion"])
    except (AttributeError, ValueError, KeyError, TypeError):
        return None, None
    if not 0 <= a < len(vista.mask) or not vista.mask[a]:
        return None, None
    return a, d.get("motivo")


class ClienteDQN:
    """Cliente falso: un agente entrenado haciendo de LLM (misma interfaz y mismo JSON)."""
    def __init__(self, agente, motivo="hago esta jugada porque sí"):
        self.agente, self.motivo = agente, motivo

    def __call__(self, prompt, vista, razonar=False):
        d = {"accion": int(self.agente(vista))}
        if razonar:
            d["motivo"] = self.motivo
        return json.dumps(d)


class CuotaAgotada(RuntimeError):
    """La API rechaza las llamadas por cuota (diaria o persistente tras varios reintentos): hay que parar."""


class ClienteGemini:
    """Cliente de la API REST de Gemini (sin dependencias: urllib). La clave NUNCA va en niveles.json: variable de entorno.

    pensar: presupuesto de tokens de "pensamiento" de gemini-2.5-flash (0 = sin pensar, lo más barato y rápido;
            -1 = dinámico; N = hasta N tokens). Los tokens de pensamiento cuentan como salida.
    rpm:    llamadas máximas por minuto (se espera entre llamadas para no chocar con el límite gratuito).
    """
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"

    def __init__(self, modelo="gemini-2.5-flash", api_key=None, temperatura=0.0, pensar=0, rpm=8,
                 max_salida=256, reintentos_http=5, _post=None, _dormir=time.sleep, _reloj=time.monotonic):
        self.modelo, self.temperatura, self.pensar, self.max_salida = modelo, temperatura, pensar, max_salida
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key and _post is None:
            raise RuntimeError("Falta la clave: define la variable de entorno GEMINI_API_KEY")
        self.intervalo = 60.0 / rpm if rpm else 0.0
        self.reintentos_http = reintentos_http
        self._post, self._dormir, self._reloj, self._ultima = _post or self._post_http, _dormir, _reloj, None
        self.llamadas = self.tokens_entrada = self.tokens_salida = 0

    def _post_http(self, url, cuerpo):
        req = urllib.request.Request(url, json.dumps(cuerpo).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except ValueError:
                return e.code, {}

    def _espera_por_ritmo(self):
        if self._ultima is not None and self.intervalo:
            falta = self.intervalo - (self._reloj() - self._ultima)
            if falta > 0:
                self._dormir(falta)
        self._ultima = self._reloj()

    @staticmethod
    def _retraso(cuerpo):
        """Segundos que sugiere la API en un 429 (detalles RetryInfo: "retryDelay": "12s"), o None."""
        for d in (cuerpo.get("error", {}) or {}).get("details", []):
            m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
        return None

    def __call__(self, prompt, vista=None, razonar=False):
        cuerpo = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": self.temperatura, "maxOutputTokens": self.max_salida,
                                       "responseMimeType": "application/json",
                                       "thinkingConfig": {"thinkingBudget": self.pensar}}}
        url = self.URL.format(modelo=self.modelo)
        for intento in range(self.reintentos_http + 1):
            self._espera_por_ritmo()
            self.llamadas += 1
            codigo, resp = self._post(url, cuerpo)
            if codigo == 200:
                u = resp.get("usageMetadata", {})
                self.tokens_entrada += u.get("promptTokenCount", 0)
                self.tokens_salida += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                try:
                    return resp["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    return ""                                   # sin texto (bloqueo, corte...): cuenta como respuesta inválida
            detalle = json.dumps(resp.get("error", {}))[:400]
            if codigo == 429 and "PerDay" in detalle:
                raise CuotaAgotada(f"Cuota diaria agotada ({self.modelo}): {detalle}")
            if codigo in (429, 500, 502, 503, 504) and intento < self.reintentos_http:
                self._dormir(self._retraso(resp) or min(2.0 ** (intento + 1), 60.0))
                continue
            if codigo == 429:
                raise CuotaAgotada(f"Límite de uso ({self.modelo}) tras {intento + 1} intentos: {detalle}")
            raise RuntimeError(f"Error de la API de Gemini ({codigo}): {detalle}")

    @property
    def uso(self):
        return f"{self.llamadas} llamadas, {self.tokens_entrada} tokens de entrada, {self.tokens_salida} de salida"


class AgenteLLM(Agente):
    def __init__(self, cliente, razonar=False, nombre="llm", registro=None, reintentos=1):
        self.cliente, self.razonar, self.nombre = cliente, razonar, nombre
        self.registro, self.reintentos = registro, reintentos       # registro: ruta de un .jsonl (opcional)
        self.llamadas = self.fallos = 0
        self.ultimo_motivo = None
        self._respaldo = AgenteHeuristico()

    def decidir(self, vista):
        prompt = construir_prompt(vista, self.razonar)
        accion = motivo = texto = None
        for _ in range(self.reintentos + 1):
            self.llamadas += 1
            texto = self.cliente(prompt, vista, self.razonar)
            accion, motivo = parsear_respuesta(texto, vista)
            if accion is not None:
                break
        if accion is None:                       # respuesta inválida tras los reintentos: juega el heurístico y se cuenta
            self.fallos += 1
            accion, motivo = int(self._respaldo(vista)), "(respuesta inválida: jugada de respaldo)"
        self.ultimo_motivo = motivo if self.razonar else None
        if self.registro:
            with open(self.registro, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(prompt=prompt, respuesta=texto, accion=accion, motivo=motivo), ensure_ascii=False) + "\n")
        return accion


def crear_agente_llm(cfg, device="cpu", ruta=None):
    """Agente LLM a partir de una entrada de niveles.json (tipo "llm")."""
    clase = cfg.get("cliente", "falso")
    if clase == "falso":
        from .niveles import cargar_nivel
        cliente = ClienteDQN(cargar_nivel(cfg["nivel_base"], device, ruta))
    elif clase == "gemini":
        cliente = ClienteGemini(**{k: cfg[k] for k in ("modelo", "temperatura", "pensar", "rpm", "max_salida") if k in cfg})
    else:
        raise ValueError(f"Cliente LLM desconocido: '{clase}' (disponibles: 'falso', 'gemini')")
    return AgenteLLM(cliente, razonar=bool(cfg.get("razonar", False)), registro=cfg.get("registro"))
