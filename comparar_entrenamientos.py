"""Compara varios entrenamientos: puntos medios, posición, winrate... frente al episodio.

  python comparar_entrenamientos.py --run "media=ckpt_media/metricas.jsonl" --run "mejor=ckpt_mejor" --salida plots
  python comparar_entrenamientos.py --run "media=log_media.log" --run "mejor=log_mejor.log" --hasta 7000

Cada --run es NOMBRE=RUTA, y RUTA puede ser:
  - un metricas.jsonl que escribe el entrenador (v5_0+), o la carpeta que lo contiene (CKPT_DIR): formato recomendado;
  - el log de texto de un entrenamiento anterior: se extraen las mismas tablas de evaluación y las líneas "Ep ...".

Genera en --salida un PNG por métrica (cada uno con un panel por nº de jugadores), evaluaciones.csv y una tabla
resumen por pantalla. Las barras/bandas son ±1 error típico de la media de puntos de UNA evaluación (no emparejado
entre ejecuciones): sirven de orientación, no de contraste estadístico.
"""
import argparse
import csv
import json
import os
import re
import sys

COLORES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']   # Okabe-Ito
MARCAS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
METRICAS = ('puntos', 'posicion', 'winrate', 'diferencia', 'entrenamiento')

# ------------------------------------------------------------------------------------------ lectura
RE_EVAL = re.compile(r'^== \[ep (\d+)\] 1 DQN vs (\d+) heur')
RE_IMIT = re.compile(r'^== \[Tras imitaci[oó]n: 1 DQN vs (\d+) heur')
RE_FINAL = re.compile(r'^== FINAL (\d+) jugadores')
RE_PARTIDAS = re.compile(r'^(\d+) partidas, (\d+) jugadores')
RE_FILA = re.compile(r'^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s*$')
RE_PUNTOS = re.compile(r'puntos medios DQN:\s*(-?[\d.]+)\s*±\s*([\d.]+)\s*\(heur\S*\s+(-?[\d.]+)\)')
RE_TRAIN = re.compile(r'^Ep\s+(\d+)/(\d+) \| eps ([\d.]+) \| pasos\s+(\d+) \| pts aprendiz\s+(-?[\d.]+) \| rivales\s+'
                      r'(-?[\d.]+) \| loss ([\d.]+) \| Recompensa aprendiz:\s+(-?[\d.]+) \| ([\d.]+) min')


def _fila(m):
    return dict(winrate=float(m.group(2)) / 100.0, puntos=float(m.group(3)), posicion=float(m.group(4)),
                columnas=float(m.group(5)), ilegales=int(m.group(6)))


def _leer_bloque(lineas, i):
    """Lee la tabla de evaluación que empieza tras la cabecera (lineas[i-1]). Devuelve (dict, siguiente_i)."""
    out = dict(partidas=None, dqn=None, heuristicos=None)
    j = i
    while j < len(lineas) and j < i + 8:
        t = lineas[j].strip()
        m = RE_PARTIDAS.match(t)
        if m:
            out['partidas'] = int(m.group(1))
        m = RE_FILA.match(t)
        if m:
            out['heuristicos' if m.group(1).startswith('heuristico') else 'dqn'] = _fila(m)
        m = RE_PUNTOS.search(t)
        if m:
            if out['dqn'] is not None:                       # esta línea trae 2 decimales; la tabla, 1
                out['dqn']['puntos'], out['dqn']['puntos_ee'] = float(m.group(1)), float(m.group(2))
                if out['heuristicos'] is not None:
                    out['heuristicos']['puntos'] = float(m.group(3))
            return out, j + 1
        j += 1
    return out, j


def leer_log_texto(ruta):
    lineas = open(ruta, encoding='utf-8', errors='replace').read().splitlines()
    run = dict(ruta=ruta, meta={}, evals=[], train=[])
    for i, linea in enumerate(lineas):
        if linea.startswith('Recompensa final: referencia ='):
            run['meta']['recompensa_ref'] = linea.split('=')[1].split()[0]
        elif linea.startswith('Barajas simétricas'):
            run['meta']['barajas_simetricas'] = True
        elif linea.startswith('Rivales:'):
            run['meta']['rivales'] = linea.strip()
        m = RE_TRAIN.match(linea)
        if m:
            run['train'].append(dict(episodio=int(m.group(1)), eps=float(m.group(3)), pasos=int(m.group(4)),
                                     pts_aprendiz=float(m.group(5)), pts_rivales=float(m.group(6)),
                                     loss=float(m.group(7)), minutos=float(m.group(9))))
            continue
        for rx, fase in ((RE_EVAL, 'entrenamiento'), (RE_IMIT, None), (RE_FINAL, 'final')):
            m = rx.match(linea)
            if not m:
                continue
            if rx is RE_EVAL:
                episodio, n = int(m.group(1)), int(m.group(2)) + 1
            elif rx is RE_IMIT:
                episodio, n = 0, int(m.group(1)) + 1
                # la 1ª evaluación tras imitar es tras el BC; la 2ª, tras recalibrar la escala de Q
                fase = 'tras_calibracion' if any(e['fase'] == 'tras_bc' and e['jugadores'] == n for e in run['evals']) else 'tras_bc'
            else:
                episodio, n = None, int(m.group(1))
            bloque, _ = _leer_bloque(lineas, i + 1)
            if bloque['dqn'] is None:
                break
            run['evals'].append(dict(episodio=episodio, jugadores=n, fase=fase, partidas=bloque['partidas'],
                                     dqn=bloque['dqn'], heuristicos=bloque['heuristicos'] or {}))
            break
    ultimo = max((e['episodio'] for e in run['evals'] if e['fase'] == 'entrenamiento'), default=None)
    for e in run['evals']:
        if e['fase'] == 'final':
            e['episodio'] = ultimo
    run['evals'] = [e for e in run['evals'] if e['fase'] != 'final']          # el final repite la última evaluación
    return run


def leer_jsonl(ruta):
    """metricas.jsonl del entrenador: una línea JSON por registro (tipo = meta | train | eval | imitacion).
    Si el entrenamiento se reanudó, puede haber episodios repetidos: se queda la última aparición."""
    run = dict(ruta=ruta, meta={}, evals=[], train=[])
    evals, train = {}, {}
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            r = json.loads(linea)
            t = r.get('tipo')
            if t == 'meta':
                run['meta'].update({k: v for k, v in r.items() if k != 'tipo'})
            elif t == 'train':
                train[r['episodio']] = r
            elif t == 'eval':
                evals[(r['episodio'], r['jugadores'], r.get('fase', 'entrenamiento'))] = r
            elif t == 'imitacion':
                run['meta']['imitacion'] = {k: v for k, v in r.items() if k != 'tipo'}
    run['train'] = [train[k] for k in sorted(train)]
    run['evals'] = [dict(episodio=e['episodio'], jugadores=e['jugadores'], fase=e.get('fase', 'entrenamiento'),
                         partidas=e.get('partidas'), dqn=e['dqn'], heuristicos=e.get('heuristicos', {}))
                    for _, e in sorted(evals.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    return run


def leer_run(ruta):
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, 'metricas.jsonl')
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe: {ruta}")
    return leer_jsonl(ruta) if ruta.endswith('.jsonl') else leer_log_texto(ruta)


# ------------------------------------------------------------------------------------------ series
def serie(run, n, extractor, desde=0, hasta=None, inicio=True):
    """[(episodio, valor, error|None)] de las evaluaciones del DQN con n jugadores, ordenadas por episodio."""
    pts = []
    for e in run['evals']:
        if e['jugadores'] != n or e['episodio'] is None:
            continue
        if e['fase'] == 'tras_bc' or (e['fase'] == 'tras_calibracion' and not inicio):
            continue
        if e['episodio'] < desde or (hasta is not None and e['episodio'] > hasta):
            continue
        v, ee = extractor(e)
        if v is not None:
            pts.append((e['episodio'], v, ee))
    return sorted(pts)


EXTRACTORES = {
    'puntos': (lambda e: (e['dqn'].get('puntos'), e['dqn'].get('puntos_ee')), 'Puntos medios del DQN (menos es mejor)', None),
    'posicion': (lambda e: (e['dqn'].get('posicion'), None), 'Posición media del DQN (1 = primero)', 'posicion'),
    'winrate': (lambda e: (None if e['dqn'].get('winrate') is None else 100 * e['dqn']['winrate'], None),
                'Victorias del DQN (%)', 'winrate'),
    'diferencia': (lambda e: ((e['dqn']['puntos'] - e['heuristicos']['puntos']) if e['heuristicos'].get('puntos') is not None
                              and e['dqn'].get('puntos') is not None else None, None),
                   'Puntos DQN − puntos heurísticos (menos es mejor)', 'cero'),
}


def _media_movil(vals, k):
    out = []
    for i in range(len(vals)):
        ventana = vals[max(0, i - k + 1): i + 1]
        out.append(sum(ventana) / len(ventana))
    return out


# ------------------------------------------------------------------------------------------ plots
def _estilo(plt):
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.frameon': False, 'font.size': 10})


def fig_eval(plt, runs, nombres, metrica, jugadores, args):
    extractor, ylabel, ref = EXTRACTORES[metrica]
    fig, axs = plt.subplots(1, len(jugadores), figsize=(5.6 * len(jugadores), 4.3), squeeze=False)
    for ax, n in zip(axs[0], jugadores):
        hay = False
        for k, (run, nombre) in enumerate(zip(runs, nombres)):
            s = serie(run, n, extractor, args.desde, args.hasta, not args.sin_inicio)
            if not s:
                continue
            hay = True
            x, y, ee = [p[0] for p in s], [p[1] for p in s], [p[2] for p in s]
            c, m = COLORES[k % len(COLORES)], MARCAS[k % len(MARCAS)]
            if args.suavizado > 1:
                ax.plot(x, y, marker=m, ms=3.5, lw=0.8, alpha=0.35, color=c)
                ax.plot(x, _media_movil(y, args.suavizado), lw=2.2, color=c, label=f'{nombre} (media móvil {args.suavizado})')
            else:
                ax.plot(x, y, marker=m, ms=4.5, lw=1.8, color=c, label=nombre)
                if all(e is not None for e in ee):
                    ax.fill_between(x, [a - b for a, b in zip(y, ee)], [a + b for a, b in zip(y, ee)], color=c, alpha=0.12, lw=0)
        if ref == 'posicion':
            ax.axhline((n + 1) / 2, color='gray', ls=':', lw=1)
            ax.text(0.01, (n + 1) / 2, ' azar', transform=ax.get_yaxis_transform(), va='bottom', color='gray', fontsize=9)
        elif ref == 'winrate':
            ax.axhline(100 / n, color='gray', ls=':', lw=1)
            ax.text(0.01, 100 / n, ' azar', transform=ax.get_yaxis_transform(), va='bottom', color='gray', fontsize=9)
        elif ref == 'cero':
            ax.axhline(0, color='gray', ls='--', lw=1)
            ax.text(0.01, 0, ' = heurístico', transform=ax.get_yaxis_transform(), va='bottom', color='gray', fontsize=9)
        ax.set_title(f'{n} jugadores: 1 DQN vs {n - 1} heurístico' + ('s' if n > 2 else ''))
        ax.set_xlabel('Episodio')
        ax.set_ylabel(ylabel)
        if not hay:
            ax.text(0.5, 0.5, 'sin evaluaciones', ha='center', va='center', transform=ax.transAxes, color='gray')
        elif ax is axs[0][0]:
            ax.legend(loc='best')
    fig.suptitle(ylabel + ' frente al episodio de entrenamiento', fontsize=12)
    fig.tight_layout()
    return fig


def fig_entrenamiento(plt, runs, nombres, args):
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for k, (run, nombre) in enumerate(zip(runs, nombres)):
        t = [r for r in run['train'] if r['episodio'] >= args.desde and (args.hasta is None or r['episodio'] <= args.hasta)]
        if not t:
            continue
        x = [r['episodio'] for r in t]
        c = COLORES[k % len(COLORES)]
        nan = float('nan')
        axs[0].plot(x, [nan if r.get('pts_aprendiz') is None else r['pts_aprendiz'] for r in t], color=c, lw=1.8,
                    label=f'{nombre}: aprendiz')
        if any(r.get('pts_rivales') is not None for r in t):                  # en self-play no hay rivales aparte
            axs[0].plot(x, [nan if r.get('pts_rivales') is None else r['pts_rivales'] for r in t], color=c, lw=1.0,
                        ls='--', alpha=0.7, label=f'{nombre}: rivales')
        axs[1].plot(x, [nan if r.get('loss') is None else r['loss'] for r in t], color=c, lw=1.8, label=nombre)
    axs[0].set_title('Puntos en entrenamiento (con exploración)')
    axs[0].set_ylabel('Puntos medios (ventana LOG_EVERY)')
    axs[1].set_title('Loss (no comparable si cambia la recompensa)')
    axs[1].set_ylabel('Loss')
    for ax in axs:
        ax.set_xlabel('Episodio')
    axs[0].legend(loc='best', fontsize=8)
    axs[1].legend(loc='best')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------ resumen
def resumen(runs, nombres, jugadores, args):
    print(f"\n{'ejecución':<28}{'jug':>4}{'evals':>6}{'último':>14}{'media últ.3':>13}{'mejor':>8}{'dif. vs heur (últ.3)':>22}{'1ª vez < heur':>15}")
    for run, nombre in zip(runs, nombres):
        for n in jugadores:
            ev = [e for e in run['evals'] if e['jugadores'] == n and e['fase'] == 'entrenamiento' and e['episodio'] is not None
                  and e['episodio'] >= args.desde and (args.hasta is None or e['episodio'] <= args.hasta)]
            ev.sort(key=lambda e: e['episodio'])
            if not ev:
                continue
            p = [e['dqn']['puntos'] for e in ev]
            ult = ev[-1]
            d = [e['dqn']['puntos'] - e['heuristicos']['puntos'] for e in ev if e['heuristicos'].get('puntos') is not None]
            primera = next((e['episodio'] for e in ev if e['heuristicos'].get('puntos') is not None
                            and e['dqn']['puntos'] < e['heuristicos']['puntos']), None)
            ee = ult['dqn'].get('puntos_ee')
            print(f"{nombre:<28}{n:>4}{len(ev):>6}"
                  f"{(f'{p[-1]:.2f}' + (f' ±{ee:.2f}' if ee else '')):>14}{sum(p[-3:]) / len(p[-3:]):>13.2f}{min(p):>8.2f}"
                  f"{(sum(d[-3:]) / len(d[-3:]) if d else float('nan')):>22.2f}{(str(primera) if primera is not None else '—'):>15}")


def exportar_csv(runs, nombres, ruta):
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ejecucion', 'episodio', 'fase', 'jugadores', 'partidas', 'dqn_puntos', 'dqn_puntos_ee', 'dqn_posicion',
                    'dqn_winrate', 'dqn_columnas', 'dqn_ilegales', 'heur_puntos', 'heur_posicion', 'heur_winrate', 'heur_columnas'])
        for run, nombre in zip(runs, nombres):
            for e in run['evals']:
                d, h = e['dqn'], e['heuristicos']
                w.writerow([nombre, e['episodio'], e['fase'], e['jugadores'], e['partidas'], d.get('puntos'), d.get('puntos_ee'),
                            d.get('posicion'), d.get('winrate'), d.get('columnas'), d.get('ilegales'),
                            h.get('puntos'), h.get('posicion'), h.get('winrate'), h.get('columnas')])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, metavar='NOMBRE=RUTA', help='repetible; RUTA = metricas.jsonl, carpeta o log de texto')
    ap.add_argument('--salida', default='plots_comparativa', help='carpeta de salida (PNG + evaluaciones.csv)')
    ap.add_argument('--jugadores', type=int, nargs='+', default=[4, 2], help='nº de asientos de las evaluaciones a dibujar')
    ap.add_argument('--metricas', nargs='+', choices=METRICAS, default=list(METRICAS))
    ap.add_argument('--desde', type=int, default=0, help='episodio mínimo')
    ap.add_argument('--hasta', type=int, default=None, help='episodio máximo (p. ej. para comparar con una ejecución más corta)')
    ap.add_argument('--suavizado', type=int, default=1, help='media móvil sobre K evaluaciones (1 = sin suavizar)')
    ap.add_argument('--sin-inicio', action='store_true', help='no dibujar la evaluación tras la imitación (episodio 0)')
    ap.add_argument('--formato', default='png')
    a = ap.parse_args(argv)

    nombres, runs = [], []
    for r in a.run:
        if '=' not in r:
            raise SystemExit(f"--run espera NOMBRE=RUTA, no '{r}'")
        nombre, ruta = r.split('=', 1)
        nombres.append(nombre)
        runs.append(leer_run(ruta))
        print(f"{nombre}: {len(runs[-1]['evals'])} evaluaciones, {len(runs[-1]['train'])} registros de entrenamiento  [{ruta}]")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _estilo(plt)
    os.makedirs(a.salida, exist_ok=True)
    for metrica in a.metricas:
        fig = fig_entrenamiento(plt, runs, nombres, a) if metrica == 'entrenamiento' else fig_eval(plt, runs, nombres, metrica, a.jugadores, a)
        destino = os.path.join(a.salida, f'{metrica}.{a.formato}')
        fig.savefig(destino, dpi=140)
        plt.close(fig)
        print('->', destino)
    exportar_csv(runs, nombres, os.path.join(a.salida, 'evaluaciones.csv'))
    resumen(runs, nombres, a.jugadores, a)


if __name__ == '__main__':
    main()
