# ============================================================
# SkyjoEnv.py  (v5.0)
# Novedades vs 4.1:
#  - Observación con información explícita de columnas:
#      * one-hot de la carta en mano y del descarte visible
#      * vectores "qué valores completarían una columna" (propio y rival)
#      * planos por casilla alineados con las acciones (completa columna / ayuda al rival / ganancia)
#      * ganancia máxima de colocar el descarte / la mano (lookahead de 1 paso)
#  - _conteo_restantes vectorizado (mucho más rápido)
#  - Métricas opcionales (columnas eliminadas, "regalos" al rival)
# ============================================================
import random
import numpy as np

TOTALES_INICIALES = np.array([5, 10, 15] + [10] * 12, dtype=np.float32)  # valores -2..12
N_ACCIONES = 26
N_PLANOS_PROPIO = 3
N_ESCALARES = 92   # 2 + 15 + 15 + 15 + 9 + 15 + 15 + 4 + 2
SHAPING_COEF = 0.1


def _onehot(v):
    x = np.zeros(15, dtype=np.float32)
    if v is not None:
        x[int(v) + 2] = 1.0
    return x


class SkyjoEnv:
    def __init__(self, num_jugadores=2):
        self.num_jugadores = num_jugadores
        self.metricas = False          # True solo en evaluación (cuesta algo de CPU)
        self.reset()

    # ---------------- baraja ----------------
    def _crear_baraja(self):
        baraja = [-2] * 5 + [-1] * 10 + [0] * 15
        for v in range(1, 13):
            baraja += [v] * 10
        random.shuffle(baraja)
        return baraja

    def reset(self):
        self.mazo = self._crear_baraja()
        self.descartes = []
        self.tableros = np.zeros((self.num_jugadores, 3, 4, 2), dtype=np.int32)  # [...,0]=valor [...,1]=vis (-1 quitada, 0 oculta, 1 visible)
        for j in range(self.num_jugadores):
            for f in range(3):
                for c in range(4):
                    self.tableros[j, f, c, 0] = self.mazo.pop()
            for celda in random.sample(range(12), 2):
                self.tableros[j, celda // 4, celda % 4, 1] = 1
        self.descartes.append(self.mazo.pop())
        self.turno_actual = 0
        self.fase_turno = 1
        self.carta_en_mano = None
        self.origen_robo = None
        self.cerrador_id = None
        self.turnos_restantes = -1
        self.game_over = False
        self.columnas_eliminadas = [0] * self.num_jugadores
        self.regalos = [0] * self.num_jugadores
        return self.obtener_observacion(self.turno_actual)

    def _reciclar_descartes_si_es_necesario(self):
        if len(self.mazo) == 0 and len(self.descartes) > 1:
            top = self.descartes.pop()
            self.mazo = self.descartes[:]
            random.shuffle(self.mazo)
            self.descartes = [top]

    # ---------------- información pública ----------------
    def _conteo_restantes(self):
        """Cartas NO vistas (mazo + ocultas de todos) por valor -2..12."""
        vistas = np.zeros(15, dtype=np.float32)
        if self.descartes:
            vistas += np.bincount(np.asarray(self.descartes, dtype=np.int64) + 2, minlength=15)[:15].astype(np.float32)
        for j in range(self.num_jugadores):
            t = self.tableros[j]
            vis = t[:, :, 0][t[:, :, 1] == 1]
            if vis.size:
                vistas += np.bincount(vis.astype(np.int64) + 2, minlength=15)[:15].astype(np.float32)
        if self.carta_en_mano is not None:
            vistas[int(self.carta_en_mano) + 2] += 1.0
        return np.maximum(TOTALES_INICIALES - vistas, 0.0)

    def _valor_esperado_oculta(self, conteo=None):
        if conteo is None:
            conteo = self._conteo_restantes()
        total = conteo.sum()
        if total <= 0:
            return 5.0
        return float((conteo * np.arange(-2, 13)).sum() / total)

    def _puntos_visibles(self, j):
        t = self.tableros[j]
        return float(t[:, :, 0][t[:, :, 1] == 1].sum())

    def _num_ocultas(self, j):
        return int((self.tableros[j][:, :, 1] == 0).sum())

    def _puntos_estimados(self, j, e_oculta):
        return self._puntos_visibles(j) + self._num_ocultas(j) * e_oculta

    def _potencial(self, j):
        return -self._puntos_estimados(j, self._valor_esperado_oculta())

    # ---------------- columnas ----------------
    def _valores_que_completan(self, j):
        """Vector (15): 1 en cada valor v que eliminaría alguna columna del jugador j
        (columna con exactamente dos visibles iguales a v; la tercera se puede reemplazar)."""
        t = self.tableros[j]
        need = np.zeros(15, dtype=np.float32)
        for c in range(4):
            vis, vals = t[:, c, 1], t[:, c, 0]
            if np.any(vis == -1):
                continue
            for v in set(vals[vis == 1].tolist()):
                if np.sum((vis == 1) & (vals == v)) == 2:
                    need[int(v) + 2] = 1.0
        return need

    def _analisis_colocar(self, j, carta, e_oculta):
        """Para colocar 'carta' en cada casilla del jugador j devuelve
        (completa[3,4], ganancia_en_puntos[3,4], valida[3,4])."""
        completa = np.zeros((3, 4), dtype=np.float32)
        ganancia = np.zeros((3, 4), dtype=np.float32)
        valida = np.zeros((3, 4), dtype=bool)
        if carta is None:
            return completa, ganancia, valida
        t = self.tableros[j]
        for c in range(4):
            vis, vals = t[:, c, 1], t[:, c, 0]
            if vis[0] == -1:
                continue
            for r in range(3):
                valida[r, c] = True
                old = float(vals[r]) if vis[r] == 1 else e_oculta
                otros = [k for k in range(3) if k != r]
                if all(vis[k] == 1 and vals[k] == carta for k in otros):
                    completa[r, c] = 1.0
                    ganancia[r, c] = 2 * carta + old   # desaparece toda la columna
                else:
                    ganancia[r, c] = old - carta
        return completa, ganancia, valida

    # ---------------- observación ----------------
    def obtener_observacion(self, jugador_id=None):
        if jugador_id is None:
            jugador_id = self.turno_actual
        n = self.num_jugadores
        idx = [(jugador_id + i) % n for i in range(n)]       # el propio siempre en 0
        tr = self.tableros[idx]
        visible = (tr[:, :, :, 1] == 1)
        valores = np.where(visible, (tr[:, :, :, 0] - 5.0) / 7.0, 0.0).astype(np.float32)
        vis = tr[:, :, :, 1].astype(np.float32)

        conteo = self._conteo_restantes()
        e_oculta = self._valor_esperado_oculta(conteo)
        rivales = idx[1:]

        need_propio = self._valores_que_completan(jugador_id)
        need_rival = np.zeros(15, dtype=np.float32)
        for r in rivales:
            need_rival = np.maximum(need_rival, self._valores_que_completan(r))

        top = int(self.descartes[-1]) if self.descartes else None
        mano = None if self.carta_en_mano is None else int(self.carta_en_mano)

        comp_top, gan_top, valida = self._analisis_colocar(jugador_id, top, e_oculta)
        if mano is not None:
            comp_ref, gan_ref, _ = self._analisis_colocar(jugador_id, mano, e_oculta)
        else:
            comp_ref, gan_ref = comp_top, gan_top      # en fase 1 la referencia es el descarte

        t = self.tableros[jugador_id]
        ayuda = ((t[:, :, 1] == 1) & (need_rival[(t[:, :, 0] + 2) % 15] > 0)).astype(np.float32)
        planos = np.stack([comp_ref, ayuda, np.clip(gan_ref / 12.0, -1.5, 3.0)]).astype(np.float32)

        def flag(need, v):
            return float(need[v + 2]) if v is not None else 0.0

        def gmax(gan):
            return float(np.clip(gan[valida].max() / 20.0, -2, 2)) if valida.any() else 0.0

        ocultas_propias = self._num_ocultas(jugador_id)
        ocultas_riv = [self._num_ocultas(r) for r in rivales]
        est_propio = self._puntos_estimados(jugador_id, e_oculta)
        est_riv = [self._puntos_estimados(r, e_oculta) for r in rivales]

        extra = np.array([
            1.0 if self.cerrador_id is not None else 0.0,
            max(self.turnos_restantes, 0) / 3.0,
            ocultas_propias / 12.0,
            float(np.mean(ocultas_riv)) / 12.0,
            float(np.min(ocultas_riv)) / 12.0,
            self._puntos_visibles(jugador_id) / 50.0,
            est_propio / 50.0,
            float(np.clip((min(est_riv) - est_propio) / 30.0, -2, 2)),
            len(self.mazo) / 150.0,
        ], dtype=np.float32)

        fase_n = self.fase_turno / 2.0
        origen_flag = 0.0 if self.origen_robo in (None, 'MAZO') else 1.0
        escalares = np.concatenate([
            np.array([fase_n, origen_flag], dtype=np.float32),               # 2
            _onehot(mano), _onehot(top),                                      # 30
            conteo / TOTALES_INICIALES,                                       # 15
            extra,                                                            # 9
            need_propio, need_rival,                                          # 30
            np.array([flag(need_propio, top), flag(need_rival, top),
                      flag(need_propio, mano), flag(need_rival, mano),
                      gmax(gan_top), gmax(gan_ref) if mano is not None else 0.0],
                     dtype=np.float32),                                       # 6
        ]).astype(np.float32)
        assert escalares.shape[0] == N_ESCALARES

        return {
            'valores_tableros': valores,
            'visibilidad_tableros': vis,
            'planos_propio': planos,
            'escalares': escalares,
        }

    # ---------------- reglas ----------------
    def get_action_mask(self, jugador_id=None):
        if jugador_id is None:
            jugador_id = self.turno_actual
        mask = np.zeros(N_ACCIONES, dtype=bool)
        t = self.tableros[jugador_id]
        if self.fase_turno == 1:
            mask[0] = True
            mask[1] = True
        else:
            for i in range(12):
                f, c = divmod(i, 4)
                if t[f, c, 1] != -1:
                    mask[2 + i] = True
                if self.origen_robo == 'MAZO' and t[f, c, 1] == 0:
                    mask[14 + i] = True
        return mask

    def procesar_triples_columna(self, jugador_id, contar=True):
        t = self.tableros[jugador_id]
        for c in range(4):
            if np.all(t[:, c, 1] == 1) and t[0, c, 0] == t[1, c, 0] == t[2, c, 0]:
                for r in range(3):
                    self.descartes.append(int(t[r, c, 0]))
                    t[r, c, 0] = 0
                    t[r, c, 1] = -1
                if contar:
                    self.columnas_eliminadas[jugador_id] += 1

    def step(self, accion):
        if self.game_over:
            raise ValueError("La partida ha terminado. Llama a reset().")
        j = self.turno_actual
        t = self.tableros[j]
        if not self.get_action_mask(j)[accion]:
            return self.obtener_observacion(j), -10.0, False, {"error": "Acción ilegal"}
        reward = 0.0

        if self.fase_turno == 1:
            if accion == 0:
                self.carta_en_mano = int(self.descartes.pop())
                self.origen_robo = 'DESCARTE'
            else:
                self._reciclar_descartes_si_es_necesario()
                self.carta_en_mano = int(self.mazo.pop())
                self.origen_robo = 'MAZO'
            self.fase_turno = 2
            return self.obtener_observacion(j), 0.0, False, {}

        # Fase 2
        pot_antes = self._potencial(j)
        if 2 <= accion <= 13:
            f, c = divmod(accion - 2, 4)
            if t[f, c, 1] != -1:
                self.descartes.append(int(t[f, c, 0]))
            t[f, c, 0] = self.carta_en_mano
            t[f, c, 1] = 1
        else:
            self.descartes.append(int(self.carta_en_mano))
            f, c = divmod(accion - 14, 4)
            t[f, c, 1] = 1
        self.carta_en_mano = None
        self.origen_robo = None
        self.procesar_triples_columna(j)

        if self.metricas and len(self.descartes) > 0:
            top = int(self.descartes[-1])
            for r in range(self.num_jugadores):
                if r != j and self._valores_que_completan(r)[top + 2] > 0:
                    self.regalos[j] += 1
                    break

        reward = SHAPING_COEF * (self._potencial(j) - pot_antes)

        # cierre de partida
        if self.cerrador_id is None and self._num_ocultas(j) == 0:
            self.cerrador_id = j
            self.turnos_restantes = self.num_jugadores - 1
        elif self.turnos_restantes > 0:
            self.turnos_restantes -= 1
            if self.turnos_restantes == 0:
                self.game_over = True

        self.fase_turno = 1
        self.turno_actual = (j + 1) % self.num_jugadores
        return self.obtener_observacion(self.turno_actual), reward, self.game_over, {}

    def calcular_puntuaciones_finales(self):
        puntos = []
        for j in range(self.num_jugadores):
            self.tableros[j][:, :, 1] = np.where(self.tableros[j][:, :, 1] == 0, 1, self.tableros[j][:, :, 1])
            self.procesar_triples_columna(j, contar=False)
            t = self.tableros[j]
            puntos.append(int(t[:, :, 0][t[:, :, 1] == 1].sum()))
        if self.cerrador_id is not None:
            otros = [p for k, p in enumerate(puntos) if k != self.cerrador_id]
            if min(otros) <= puntos[self.cerrador_id] and puntos[self.cerrador_id] > 0:
                puntos[self.cerrador_id] *= 2
        return puntos
