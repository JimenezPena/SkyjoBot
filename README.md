DQN que aprende a jugar a Skyjo™ y servidor web para jugar humano contra IA. Pruébalo en: skyjobot.onrender.com

## Estructura

```
skyjo/
  heuristico/   baseline sin pesos (funciona con cualquier nº de jugadores y cualquier versión del entorno)
  v4_2/         Dueling Double DQN, 26 acciones (modelo 4.1 + heurístico y duelos cada 2500 episodios)
  v5_0/         observación y red independientes del nº de jugadores (2-8), desde la vista pública (en desarrollo: aún sin entrenador)
  niveles.py    registro de niveles: cargar_nivel("nombre") -> agente válido con el entorno de cualquier versión
  vista.py      vista pública de un asiento (solo información visible) y adaptadores
  agentes.py    interfaz de agente agente(vista)->acción: humano, ruido epsilon
  partida.py    jugar_partida / evaluar con 2-8 asientos (mezclas de humanos, modelos y heurístico)
  consola.py    partida por consola: python -m skyjo.consola humano heuristico ...
niveles.json    niveles disponibles (heuristico, dqn_4.2, aleatorio, ...)
modelos/        pesos finales que usan los niveles
online_app.py + templates/index.html   la web
train.py / evaluate.py                 entrenamiento y evaluación por línea de comandos
notebooks/colab_train.ipynb            entrenar en Google Colab (clona el repo y guarda todo en Drive)
tests/                                 la web y los modelos deben pasar siempre (también en GitHub Actions)
PreviousVersions/                      pesos antiguos, incompatibles con el código actual
```

Cada carpeta de `skyjo/` es autocontenida (entorno + observación + red + buffer + entrenamiento + agentes) para poder
comparar qué decisiones de diseño y qué información de entrada hacen que una red juegue mejor o peor.

## Entrenar (Colab o local)

```bash
python train.py --version v4_2 --out-dir /content/drive/MyDrive/skyjo     # checkpoints y modelo en Drive
python train.py --version v4_2 --set NUM_EPISODIOS=5000 LR=3e-4           # sobrescribe hiperparámetros
python evaluate.py torneo --version v4_2 --ckpt-dir <carpeta_checkpoints> # cada checkpoint vs heurístico
python evaluate.py niveles -n 300                                         # todos los niveles entre sí
```

Cada entrenamiento escribe `<modelo>.meta.json` (versión, commit, CONFIG y resultado vs heurístico). Para publicar un
modelo como nivel: copiar el `.pth` a `modelos/` y añadir una entrada a `niveles.json`:

```json
"dificil": {"tipo": "dqn", "version": "v4_2", "modelo": "modelos/v4_2_best.pth", "epsilon": 0.0}
```

## Web

El oponente por defecto es `dqn_4.2`; se cambia con la variable de entorno `SKYJO_NIVEL` o por partida con
`POST /api/reset {"nivel": "heuristico"}` (opcional; `GET /api/niveles` lista los disponibles).
`python -m pytest -q tests` juega partidas completas por la API con cada nivel y comprueba que `index.html` solo usa
rutas que existen y que no se filtran cartas ocultas.

## Vista pública y partidas con N jugadores

Cada agente decide a partir de una `Vista` (`skyjo/vista.py`): sus cartas y las visibles de los demás, con los tableros
girados al asiento del agente (0 = yo, 1 = izquierda, que recibe mis descartes, n-1 = derecha, de quien viene el
descarte), historial de descartes, nº de jugadores y cartas ocultas. Nunca contiene cartas ocultas ajenas.
La observación del modelo 4.2 se construye solo desde la vista (idéntica a la anterior). El DQN 4.2 solo juega con 2
asientos; el heurístico y el aleatorio, con 2-8.

```bash
python -m skyjo.consola humano heuristico heuristico heuristico
python evaluate.py partida heuristico dqn_4.2 -n 300
```

## Historial de versiones

heuristico: bot sencillo (coge el descarte si ahorra >= 3 puntos esperados, coloca donde más gana y, si la carta es del mazo y gana < 2, la descarta y destapa una oculta). Referencia para medir el resto.

skyjo_dqn_model_5.0 (en desarrollo): observación y red independientes del nº de jugadores, construidas solo desde la vista pública (tablero propio, izquierda y derecha en ranuras dedicadas, resto de rivales resumidos con media y máximo). La 5.0 anterior (columnas explícitas, n-step, soft update) nunca aprendió y se retiró; sigue en la rama/tag de respaldo anterior al cambio.

skyjo_dqn_model_4.2 (web, ahora modelos/v4_2.pth): mismo modelo que 4.1; añade evaluación contra el heurístico cada 2500 episodios y checkpoints.

skyjo_dqn_model_4.1.pth: Dueling DQN (ramas V(s) y A(s,a)) sobre Double DQN, 26 acciones.

skyjo_dqn_model_4.0.pth: Increases the number of actions to 26, such that when the card from the deck is discarded, the dqn (and the human player) can choose which of the unrevealed cards to flip, instead of automatically flipping the first unrevealed card.

skyjo_dqn_model_3.0.pth: Solves a feature leak that allowed to see unrevealead cards. Adds aditional observables, like number of own unrevelaed cards, the lowest number of unrevealed cards of any player, self visible points, estimation of self total points, comparison with other players, etc

skyjo_dqn_model_2.0.pth: First version uploaded to Git. Has 15 actions available: pick from discard pile, pick from the deck, replace those with one of the 12 cards, and if pick from deck, discard and reveal the first unrevealed card.
