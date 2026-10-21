"""Jugar en la terminal con asientos mixtos: humanos, modelos (niveles.json), heurístico, aleatorio.

  python -m skyjo.consola humano heuristico dqn_4.2          # 3 asientos (el dqn_4.2 solo sabe jugar a 2)
  python -m skyjo.consola humano heuristico aleatorio heuristico
Cada argumento es un asiento: 'humano' o el nombre de un nivel de niveles.json.
"""
import argparse

from .agentes import Agente
from .partida import jugar_partida


def _celda(v, e):
    return " --" if e == -1 else ("  ?" if e == 0 else f"{int(v):>3}")


def _etiqueta(i, n):
    if i == 0:
        return "Tú"
    if n == 2:
        return "Rival (izquierda y derecha)"
    if i == 1:
        return "Izquierda (recibe tus descartes)"
    if i == n - 1:
        return "Derecha (te pasa sus descartes)"
    return f"Jugador +{i}"


def vista_a_texto(vista):
    out = []
    for i in range(vista.n):
        out.append(f"{_etiqueta(i, vista.n)}   [ocultas: {int((vista.estado[i] == 0).sum())}]")
        for f in range(3):
            out.append("  " + " ".join(_celda(vista.valores[i, f, c], vista.estado[i, f, c]) for c in range(4)))
    out.append(f"Descarte visible: {vista.top_descarte} | mazo: {vista.n_mazo} cartas")
    if vista.mano is not None:
        out.append(f"Tienes en la mano: {vista.mano} ({'del mazo' if vista.origen == 'MAZO' else 'del descarte'})")
    if vista.cerrador is not None:
        out.append("¡Alguien ha cerrado la ronda!")
    return "\n".join(out)


def describir_accion(a, vista):
    if a == 0:
        return f"coge el descarte ({vista.top_descarte})"
    if a == 1:
        return "roba del mazo"
    f, c = divmod((a - 2) % 12, 4)
    if 2 <= a <= 13:
        return f"coloca la {vista.mano} en fila {f + 1}, col {c + 1}"
    return f"descarta la {vista.mano} y destapa fila {f + 1}, col {c + 1}"


class AgenteConsola(Agente):
    def __init__(self, nombre="humano", entrada=input, salida=print):
        self.nombre, self.entrada, self.salida = nombre, entrada, salida

    @staticmethod
    def _parsear(txt):
        try:
            p = txt.strip().lower().split()
            if len(p) == 1:
                return int(p[0])
            if len(p) == 3 and p[0] in ('p', 'd'):
                f, c = int(p[1]) - 1, int(p[2]) - 1
                if 0 <= f < 3 and 0 <= c < 4:
                    return (2 if p[0] == 'p' else 14) + f * 4 + c
        except ValueError:
            pass
        return -1

    def decidir(self, vista):
        self.salida("\n" + vista_a_texto(vista))
        if vista.fase == 1:
            self.salida("0) coger el descarte    1) robar del mazo")
        else:
            self.salida("p FILA COL) colocar la carta de la mano" + ("   d FILA COL) descartarla y destapar" if vista.origen == 'MAZO' else ""))
        legales = {int(a) for a in vista.acciones_legales()}
        while True:
            a = self._parsear(self.entrada("> "))
            if a in legales:
                return a
            self.salida("Acción no válida, prueba otra vez.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('asientos', nargs='+', help="'humano' o nombre de un nivel de niveles.json")
    ap.add_argument('--razonar', action='store_true', help='los asientos LLM explican cada jugada (gasta tokens de salida)')
    ap.add_argument('--semilla', type=int, default=None, help='mismo reparto en cada ejecución con la misma semilla')
    args = ap.parse_args(argv)
    from .niveles import cargar_nivel
    agentes = [AgenteConsola(f"humano{i + 1}") if s == 'humano' else cargar_nivel(s) for i, s in enumerate(args.asientos)]

    if args.razonar:
        for a in agentes:
            if hasattr(getattr(a, 'base', a), 'razonar'):
                getattr(a, 'base', a).razonar = True

    def on_paso(asiento, agente, vista, accion):
        if not isinstance(agente, AgenteConsola):
            print(f"[asiento {asiento + 1}: {agente.nombre}] {describir_accion(accion, vista)}")
            motivo = getattr(getattr(agente, 'base', agente), 'ultimo_motivo', None)
            if motivo:
                print(f"    ↳ {motivo}")
    r = jugar_partida(agentes, on_paso=on_paso, semilla=args.semilla)
    print("\nPuntos finales: " + ", ".join(f"{a.nombre}={p}" for a, p in zip(agentes, r.puntos)))
    print("Ganan: " + ", ".join(agentes[k].nombre for k in r.ganadores))


if __name__ == "__main__":
    main()
