import random
import numpy as np

# ---------------------------------------------------------------------------
# Constantes globales (v4.1)
# ---------------------------------------------------------------------------
VALORES_CARTA = np.arange(-2, 13, dtype=np.float32)                 # -2 ... 12
TOTALES_INICIALES = np.array([5, 10, 15] + [10] * 12, dtype=np.float32)
N_ESCALARES = 28       # 4 básicos + 15 conteo + 9 nuevas features
N_ACCIONES = 26        # 0-1 robar | 2-13 colocar en casilla | 14-25 descartar (mazo) y destapar casilla
SHAPING_COEF = 0.1     # escala de la recompensa intermedia basada en potencial

class SkyjoEnv:
    def __init__(self, num_jugadores=2, semilla=None):
        self.num_jugadores = num_jugadores
        # sin semilla se usa el generador global (random.seed sigue funcionando); con semilla, uno propio de la partida
        self._rng = None if semilla is None else random.Random(semilla)
        self.reset()

    @property
    def rng(self):
        return self._rng or random          # (el módulo `random` no se puede copiar con deepcopy, por eso no se guarda)

    def _crear_baraja(self):
        baraja = (
            [-2] * 5 +
            [-1] * 10 +
            [0] * 15 +
            [v for v in range(1, 13) for _ in range(10)]
        )
        self.rng.shuffle(baraja)
        return baraja

    def reset(self):
        self.mazo = self._crear_baraja()
        self.descartes = []

        # Tableros: Shape (num_jugadores, 3, 4, 2)
        # Canal 0: Valor | Canal 1: Visibilidad (1=Visible, 0=Oculta, -1=Eliminada)
        self.tableros = np.zeros((self.num_jugadores, 3, 4, 2), dtype=int)

        for j in range(self.num_jugadores):
            for f in range(3):
                for c in range(4):
                    self.tableros[j, f, c, 0] = self.mazo.pop()
                    self.tableros[j, f, c, 1] = 0

            posiciones = [(r, c) for r in range(3) for c in range(4)]
            reveladas_inicio = self.rng.sample(posiciones, 2)
            for r, c in reveladas_inicio:
                self.tableros[j, r, c, 1] = 1

        self.descartes.append(self.mazo.pop())

        self.turno_actual = 0
        self.cerrador_id = None
        self.turnos_restantes = -1
        self.game_over = False

        self.fase_turno = 1
        self.carta_en_mano = None
        self.origen_robo = None

        return self.obtener_observacion(self.turno_actual)

    def _calcular_puntos_tablero(self, jugador_id):
        """Suma el valor de las cartas VISIBLES en el tablero del jugador."""
        tablero = self.tableros[jugador_id]
        mask_visibles = (tablero[:, :, 1] == 1)
        return float(np.sum(tablero[:, :, 0][mask_visibles]))

    def procesar_triples_columna(self, jugador_id):
        tablero = self.tableros[jugador_id]
        for c in range(4):
            if np.all(tablero[:, c, 1] == 1):
                val0, val1, val2 = tablero[0, c, 0], tablero[1, c, 0], tablero[2, c, 0]
                if val0 == val1 == val2:
                    for r in range(3):
                        self.descartes.append(tablero[r, c, 0])
                        tablero[r, c, 0] = 0
                        tablero[r, c, 1] = -1

    def _conteo_restantes(self):
        """Nº de cartas de cada valor que AÚN NO se han visto (mazo + boca abajo)."""
        vistas = np.zeros(15, dtype=np.float32)
        for carta in self.descartes:
            vistas[int(carta) + 2] += 1.0
        for j in range(self.num_jugadores):
            t = self.tableros[j]
            for carta in t[:, :, 0][t[:, :, 1] == 1]:
                vistas[int(carta) + 2] += 1.0
        if self.carta_en_mano is not None:
            vistas[int(self.carta_en_mano) + 2] += 1.0
        return np.maximum(TOTALES_INICIALES - vistas, 0.0)

    def obtener_vector_conteo(self):
        """Proporción de cartas de cada valor aún no reveladas (15 elementos)."""
        return self._conteo_restantes() / TOTALES_INICIALES

    def _valor_esperado_oculta(self, conteo=None):
        """Valor medio esperado de una carta desconocida, según lo que aún no se ha visto."""
        if conteo is None:
            conteo = self._conteo_restantes()
        total = float(conteo.sum())
        if total <= 0:
            return 5.0
        return float((conteo * VALORES_CARTA).sum() / total)

    def _puntos_estimados(self, jugador_id, valor_esperado):
        """Suma visible + (nº ocultas x valor esperado). Solo usa información pública."""
        t = self.tableros[jugador_id]
        visibles = float(t[:, :, 0][t[:, :, 1] == 1].sum())
        n_ocultas = int(np.sum(t[:, :, 1] == 0))
        return visibles + n_ocultas * valor_esperado

    def _potencial(self, jugador_id):
        """Potencial para reward shaping: menos puntos estimados = mejor."""
        return -self._puntos_estimados(jugador_id, self._valor_esperado_oculta())

    def obtener_observacion(self, jugador_id=None):
        """Observación del modelo: se calcula SOLO desde la vista pública (skyjo.vista), igual que al jugar."""
        from skyjo.vista import construir_vista            # imports locales: evitan el ciclo env <-> observacion
        from .observacion import observacion_desde_vista
        return observacion_desde_vista(construir_vista(self, jugador_id))

    def _reciclar_descartes_si_es_necesario(self):
        if len(self.mazo) == 0 and len(self.descartes) > 1:
            top_carta = self.descartes.pop()
            self.mazo = self.descartes[:]
            self.rng.shuffle(self.mazo)
            self.descartes = [top_carta]

    def calcular_puntuaciones_finales(self):
        """Calcula los puntos finales oficiales de Skyjo."""
        for j in range(self.num_jugadores):
            self.tableros[j, :, :, 1] = np.where(
                self.tableros[j, :, :, 1] == 0, 1, self.tableros[j, :, :, 1]
            )
            self.procesar_triples_columna(j)

        puntos = [
            int(np.sum(np.where(self.tableros[i, :, :, 1] != -1, self.tableros[i, :, :, 0], 0)))
            for i in range(self.num_jugadores)
        ]

        if self.cerrador_id is not None:
            puntos_cerrador = puntos[self.cerrador_id]
            otros_puntos = [puntos[i] for i in range(self.num_jugadores) if i != self.cerrador_id]
            if min(otros_puntos) <= puntos_cerrador and puntos_cerrador > 0:
                puntos[self.cerrador_id] *= 2

        return puntos

    def step(self, action):
        if self.game_over:
            raise ValueError("La partida ha terminado. Llama a reset().")

        j_id = self.turno_actual
        tablero = self.tableros[j_id]

        mask = self.get_action_mask(j_id)
        if not mask[action]:
            return self.obtener_observacion(j_id), -10.0, False, {"error": "Acción ilegal"}

        recompensa = 0.0

        # FASE 1
        if self.fase_turno == 1:
            if action == 0:
                self.carta_en_mano = self.descartes.pop()
                self.origen_robo = 'DESCARTE'
            elif action == 1:
                self._reciclar_descartes_si_es_necesario()
                self.carta_en_mano = self.mazo.pop()
                self.origen_robo = 'MAZO'

            self.fase_turno = 2
            return self.obtener_observacion(j_id), 0.0, False, {"fase": 2}

        # FASE 2
        elif self.fase_turno == 2:
            carta_robada = self.carta_en_mano
            potencial_antes = self._potencial(j_id)

            if 2 <= action <= 13:
                cell_idx = action - 2
                f, c = divmod(cell_idx, 4)
                valor_anterior = tablero[f, c, 0]
                visibilidad_anterior = tablero[f, c, 1]

                tablero[f, c, 0] = carta_robada
                tablero[f, c, 1] = 1

                if visibilidad_anterior != -1:
                    self.descartes.append(valor_anterior)

            elif 14 <= action <= 25 and self.origen_robo == 'MAZO':
                cell_idx = action - 14
                f, c = divmod(cell_idx, 4)
                self.descartes.append(carta_robada)
                tablero[f, c, 1] = 1

            self.procesar_triples_columna(j_id)

            self.carta_en_mano = None
            self.origen_robo = None

            recompensa += SHAPING_COEF * (self._potencial(j_id) - potencial_antes)

            cartas_ocultas = np.sum(tablero[:, :, 1] == 0)

            if self.cerrador_id is None and cartas_ocultas == 0:
                self.cerrador_id = j_id
                self.turnos_restantes = self.num_jugadores - 1
            elif self.turnos_restantes > 0:
                self.turnos_restantes -= 1

            if self.turnos_restantes == 0:
                self.game_over = True

            self.fase_turno = 1
            siguiente_jugador = (self.turno_actual + 1) % self.num_jugadores
            self.turno_actual = siguiente_jugador

            return self.obtener_observacion(siguiente_jugador), recompensa, self.game_over, {"fase": 1}

    def get_action_mask(self, jugador_id=None):
        if jugador_id is None:
            jugador_id = self.turno_actual

        mask = np.zeros(N_ACCIONES, dtype=bool)
        tablero = self.tableros[jugador_id]

        if self.fase_turno == 1:
            mask[0] = True
            mask[1] = True
            return mask

        elif self.fase_turno == 2:
            for cell_idx in range(12):
                r, c = divmod(cell_idx, 4)
                if tablero[r, c, 1] != -1:
                    mask[cell_idx + 2] = True

            if self.origen_robo == 'MAZO':
                for cell_idx in range(12):
                    r, c = divmod(cell_idx, 4)
                    if tablero[r, c, 1] == 0:
                        mask[cell_idx + 14] = True

            return mask
