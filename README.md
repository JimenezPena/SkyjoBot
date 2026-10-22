DQN que aprende a jugar a Skyjo™ y servidor web para jugar humano contra IA. Pruébalo en: skyjobot.onrender.com

## Estructura

```
skyjo/
  heuristico/   baseline sin pesos (funciona con cualquier nº de jugadores y cualquier versión del entorno)
  v4_2/         Dueling Double DQN, 26 acciones (modelo 4.1 + heurístico y duelos cada 2500 episodios)
  v5_0/         observación y red independientes del nº de jugadores (2-8), desde la vista pública; entrenamiento self-play de
                4 jugadores, con imitación previa del heurístico (imitacion.py), pérdida de margen DQfD y checkpoints reanudables
  niveles.py    registro de niveles: cargar_nivel("nombre") -> agente válido con el entorno de cualquier versión
  vista.py      vista pública de un asiento (solo información visible) y adaptadores
  agentes.py    interfaz de agente agente(vista)->acción: humano, ruido epsilon
  partida.py    jugar_partida / evaluar con 2-8 asientos (mezclas de humanos, modelos y heurístico)
  consola.py    partida por consola: python -m skyjo.consola humano heuristico ...
niveles.json    niveles disponibles (heuristico, dqn_4.2, aleatorio, ...)
modelos/        pesos finales que usan los niveles (v4_1, v4_2, v5_0, v5_1)
online_app.py + templates/        la web: index.html (1 vs IA), cuatro.html (1 vs 3 bots), multijugador.html (2 humanos)
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
python train.py --version v5_0 --out-dir /content/drive/MyDrive/skyjo     # self-play de 4 jugadores (--set JUGADORES=2 reproduce 4.2)
python train.py --version v4_2 --set NUM_EPISODIOS=5000 LR=3e-4           # sobrescribe hiperparámetros
python evaluate.py torneo --version v4_2 --ckpt-dir <carpeta_checkpoints> # cada checkpoint vs heurístico
python evaluate.py niveles -n 300                                         # todos los niveles entre sí
```

Cada entrenamiento escribe `<modelo>.meta.json` (versión, commit, CONFIG y resultado vs heurístico). Para publicar un
modelo como nivel: copiar el `.pth` a `modelos/` y añadir una entrada a `niveles.json`:

```json
"dificil": {"tipo": "dqn", "version": "v4_2", "modelo": "modelos/v4_2_best.pth", "epsilon": 0.0}
```

### Imitación del heurístico + DQfD (v5_0)

Con `IMIT_EPISODIOS > 0` el entrenamiento no empieza de cero (`skyjo/v5_0/imitacion.py`):

1. Se juegan `IMIT_EPISODIOS` partidas heurístico vs heurístico (con los mismos `JUGADORES` asientos del entrenamiento). Sus
   transiciones, con las mismas recompensas que el entrenamiento normal, llenan el replay buffer (idea central de DQfD).
2. Behaviour cloning: entropía cruzada sobre las acciones del heurístico, con las ilegales enmascaradas
   (`IMIT_EPOCHS`, `IMIT_BATCH`, `IMIT_LR`). La exactitud de imitación aparece en el log (`[BC] epoch ...`).
3. Recalibración de la escala de Q: `CALIBRATION_PRESTEPS` pasos de DQN sobre las demos, para que los valores Q tengan
   la escala de las recompensas antes de explorar. Después empieza el DQN normal con `EPSILON_START_IMIT`.
4. Pérdida de margen DQfD (opcional, `DEMO_LAMBDA > 0`): durante los primeros `DEMO_STEPS` pasos se añade a cada
   actualización `mean(relu(DEMO_MARGIN - (Q(s,a_demo) - max Q(s,otras))))` sobre un batch de `DEMO_BATCH` demos, para no
   olvidar la imitación mientras la red aprende por refuerzo.

### Checkpoints y reanudación (Colab)

El acceso a GPU en Colab es limitado, así que el entrenamiento se puede interrumpir y retomar. Cada `SAVE_EVERY` episodios
se escribe `<CKPT_DIR>/checkpoint.pt` (redes policy y target, optimizador, paso global, episodio, epsilon, mejor winrate y el
replay buffer entero; con `BUFFER_CAPACITY` 400000 a 4 jugadores pesa del orden de medio GB). Al lanzar de nuevo `train.py` con
el mismo `CKPT_DIR` se carga y sigue en el episodio siguiente (si no hay checkpoint, empieza de cero y hace la imitación).
Cada `EVAL_EVERY` episodios se evalúa 1 DQN contra heurísticos (`EVAL_JUGADORES`; el primero decide el mejor checkpoint) y se
guardan `ep<N>.pth`, `best.pth` y `best_checkpoint.pt` (checkpoint completo del mejor momento).

### Rivales del entrenamiento (v5_0, `rivales.py`)

Con `RIVALES` definido, en cada partida aprenden `ASIENTOS_APRENDIZ` asientos elegidos al azar (solo ellos generan
transiciones y cuentan como `global_step`) y cada asiento rival se sortea según los pesos de `RIVALES`:
`heuristico` (el baseline), `actual` (la red que se entrena, greedy con `RIVAL_EPSILON` de ruido) o `checkpoint` (copia
congelada de la red: una instantánea cada `POOL_EVERY` episodios, se conservan `POOL_MAX`, más los modelos externos de
`RIVALES_CKPT`, p. ej. `['modelos/v5_1.pth']`). Sin checkpoints disponibles, `checkpoint` se sustituye por `heuristico`.
`RIVALES_FIN` interpola los pesos linealmente durante el entrenamiento (de más heurísticos a más self-play). El pool se
guarda en `<CKPT_DIR>/pool.pt` y se recupera al reanudar. `RIVALES=None` vuelve al self-play puro (todos aprenden).
Con 1 aprendiz de 4 hay ~4 veces menos transiciones por partida que en self-play: a igualdad de episodios hay ~4 veces
menos datos y pasos de gradiente (sube `NUM_EPISODIOS` o `ASIENTOS_APRENDIZ`).

## Web

El oponente por defecto es `dqn_4.2`; se cambia con la variable de entorno `SKYJO_NIVEL` o por partida con
`POST /api/reset {"nivel": "heuristico"}` (opcional; `GET /api/niveles` lista los disponibles).
**Modo 4 jugadores** (`/cuatro`): tú contra tres bots; cada bot se elige en un desplegable (`GET /api/4/niveles` lista solo los
niveles que admiten 4 asientos: el heurístico, el aleatorio y los modelos de `v5_0`; un DQN 4.2 no aparece porque solo juega
con 2). Para publicar un modelo entrenado a 4 jugadores: copiar el `.pth` a `modelos/` y añadir a `niveles.json`
`"dqn_5.0": {"tipo": "dqn", "version": "v5_0", "modelo": "modelos/v5_0.pth"}`; ya aparece en los desplegables.
Por defecto todos los bots son heurísticos (`SKYJO_NIVEL_4` lo cambia).

**Modo 2 humanos** (`/multijugador`): partida en línea entre dos personas con Flask-SocketIO. Uno crea la partida y recibe un
código de sala de 4 caracteres; el otro lo introduce para unirse. El servidor mantiene el estado (`multi_rooms`) y envía a
cada jugador su propio tablero y sus acciones legales (las cartas ocultas nunca salen del servidor). Requiere
`flask-socketio`, `gevent` y `gevent-websocket` (están en `requirements.txt`) y, en local, se arranca con `python online_app.py`
(usa `socketio.run`).

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

## Agente LLM y partidas reproducibles

`skyjo/llm.py` define `AgenteLLM`: construye un prompt con la vista pública y las acciones legales, pide una respuesta
JSON (`{"accion": n}`) y la valida (reintenta y, si sigue siendo inválida, juega el heurístico y lo cuenta en `fallos`).
El modelo es un *cliente* intercambiable `cliente(prompt, vista, razonar) -> str`; de momento solo existe el cliente
falso `llm_falso` (nivel oculto de `niveles.json`), que responde con `dqn_5.1` en el mismo formato, para probar el
circuito sin gastar cuota. `"razonar": true` (o `--razonar` en la consola) añade un campo `"motivo"` a la respuesta y
lo muestra tras cada jugada; desactivado ahorra tokens de salida. `"registro": "ruta.jsonl"` guarda cada prompt/respuesta.

**Gemini.** `llm_gemini` (nivel oculto de `niveles.json`) usa `gemini-2.5-flash` por la API REST de Google, sin
dependencias nuevas. La clave va en la variable de entorno `GEMINI_API_KEY` (nunca en el repositorio). Parámetros del
nivel: `pensar` (tokens de pensamiento; 0 = sin pensar, más barato; -1 = dinámico), `rpm` (llamadas por minuto, para
respetar el límite gratuito), `razonar`, `registro`. Reintenta 429/5xx esperando lo que indica la API, y si la cuota
diaria se agota lanza `CuotaAgotada` y la partida se detiene. La consola imprime al final llamadas, respuestas
inválidas y tokens usados. El nivel acepta además `motivo_primero` (por defecto `true`: el motivo va antes de la acción, así el razonamiento
influye en la decisión), `ayudas` (anota cada acción con los puntos/columnas que cambia, calculados en Python) y
`estrategia` (consejos de juego en el prompt); `max_salida` es el tope sin contar `pensar`. Mantenlo oculto en la web pública: cualquier visitante gastaría tu cuota.

```bash
export GEMINI_API_KEY=...
python -m skyjo.consola humano llm_gemini heuristico heuristico --razonar --semilla 3
```

Con `--semilla N` (consola) o `semilla=` en `jugar_partida`/`evaluar` el reparto es el mismo siempre, y por asiento
no depende de los agentes (comparaciones pareadas).

```bash
python -m skyjo.consola humano llm_falso heuristico heuristico --razonar --semilla 3
```

## Histogramas

Con `--plots [carpeta]` (requiere `matplotlib`, solo en `requirements-dev.txt`) se imprimen además las distribuciones
(mediana, desviación, p90, brecha al ganador, % de cierres, % de cierres doblados) y se guardan cinco histogramas:
puntos por jugador, puntos respecto al ganador, posición en el ranking, jugadas por partida y cierres.

```bash
python evaluate.py partida -n 1000 --plots plots dqn_5.1 heuristico heuristico heuristico
```

## Historial de versiones

heuristico: bot sencillo (coge el descarte si ahorra >= 3 puntos esperados, coloca donde más gana y, si la carta es del mazo y gana < 2, la descarta y destapa una oculta). Referencia para medir el resto.

skyjo_dqn_model_5.1 (modelos/v5_1.pth, nivel `dqn_5.1`): misma red y observación que la 5.0 (carpeta `v5_0`); cambia solo el entrenamiento: imitación previa del heurístico (behaviour cloning) + buffer de demos + recalibración de Q + pérdida de margen DQfD, y checkpoints reanudables. Resultado: no supera al heurístico (~18 % de victorias a 4 jugadores).

skyjo_dqn_model_5.0: observación y red independientes del nº de jugadores, construidas solo desde la vista pública (tablero propio, izquierda y derecha en ranuras dedicadas, resto de rivales resumidos con media y máximo), entrenada por self-play puro de 2 jugadores con la receta del 4.2. Resultado: no supera al heurístico (~30 % de victorias a 2 jugadores) (el 4.2 llega a ~50 %). La 5.0 anterior (columnas explícitas, n-step, soft update) nunca aprendió y se retiró; sigue en el historial de git previo a la fase 2.


skyjo_dqn_model_4.2 (web, ahora modelos/v4_2.pth): mismo modelo que 4.1; añade evaluación contra el heurístico cada 2500 episodios y checkpoints. Entrenado por self-play puro a dos jugadores: Resultado: iguala al heurístico (50.7 % vs 49.3 % de win rate a su favor, 21.7 puntos de media frente a los 24.5 del heurístico)
Resultado 2000 partidas frente a heurístico (2 jugadores):
agente                   win%  puntos  pos.  cols ilegales
heuristico               49.3    24.5  1.51  0.19        0
dqn_4.2                  50.7    21.7  1.49  0.06        0

skyjo_dqn_model_4.1.pth: Dueling DQN (ramas V(s) y A(s,a)) sobre Double DQN, 26 acciones.

skyjo_dqn_model_4.0.pth: Increases the number of actions to 26, such that when the card from the deck is discarded, the dqn (and the human player) can choose which of the unrevealed cards to flip, instead of automatically flipping the first unrevealed card.

skyjo_dqn_model_3.0.pth: Solves a feature leak that allowed to see unrevealead cards. Adds aditional observables, like number of own unrevelaed cards, the lowest number of unrevealed cards of any player, self visible points, estimation of self total points, comparison with other players, etc

skyjo_dqn_model_2.0.pth: First version uploaded to Git. Has 15 actions available: pick from discard pile, pick from the deck, replace those with one of the 12 cards, and if pick from deck, discard and reveal the first unrevealed card.
