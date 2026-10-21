"""Agente que decide con un LLM. La interfaz con el modelo es un "cliente" intercambiable:

    cliente(prompt: str, vista: Vista, razonar: bool) -> str      # respuesta JSON: {"accion": 7, "motivo": "..."}

  - ClienteDQN (en niveles.json: "cliente": "falso"): no llama a ningún LLM; juega con un nivel ya entrenado
    (p. ej. dqn_5.1) y responde con el mismo formato JSON. Sirve para probar todo el circuito sin gastar cuota.
  - Un cliente real (Gemini...) solo tiene que implementar esa misma función; el agente no cambia.

El razonamiento se activa con razonar=True (agente, "razonar": true en niveles.json o --razonar en la consola):
añade al prompt la petición de un "motivo" breve, que se guarda en agente.ultimo_motivo y en el registro.
Con razonar=False el prompt pide solo {"accion": n} y se ahorran tokens de salida.
"""
import json
import re

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
    else:
        raise ValueError(f"Cliente LLM desconocido: '{clase}' (disponible: 'falso')")
    return AgenteLLM(cliente, razonar=bool(cfg.get("razonar", False)), registro=cfg.get("registro"))
