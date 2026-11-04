"""Registro de la configuración de cada lanzamiento: <CKPT_DIR>/config.jsonl (una línea JSON por lanzamiento).

Se escribe ANTES de entrenar (también si la sesión se corta) y en modo añadir: cada reanudación suma una línea, de
modo que se ve qué configuración se usó en cada tramo. Cada línea guarda el CONFIG final (ya con los --set aplicados),
los overrides respecto al CONFIG por defecto, la línea de comandos, el commit y los cambios respecto al lanzamiento
anterior. Al reanudar, los cambios se avisan por pantalla (p. ej. si olvidas repetir un --set de recompensa).
Sin torch: se puede usar y probar sin él.
"""
import json
import os
import sys
import time

# Claves que es normal cambiar al reanudar (no se avisan como sospechosas)
ESPERABLES = {'NUM_EPISODIOS', 'SAVE_PATH', 'CKPT_DIR', 'EVAL_GAMES'}


def _plano(x):
    try:
        return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    except (TypeError, ValueError):
        return str(x)


def diferencias(a, b):
    """{clave: (valor_en_a, valor_en_b)} de las claves cuyo valor difiere (tras pasar ambos por JSON)."""
    a, b = _plano(a), _plano(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def _ultimo(ruta):
    try:
        with open(ruta, encoding='utf-8') as f:
            lineas = [l for l in f if l.strip()]
        return json.loads(lineas[-1]) if lineas else None
    except (OSError, ValueError):
        return None


def registrar(ckpt_dir, version, cfg, defecto, argv=None, commit=None):
    """Añade una línea a <ckpt_dir>/config.jsonl y devuelve (ruta, cambios_vs_anterior).

    cfg = CONFIG final; defecto = CONFIG de la versión sin tocar. Si no se puede escribir solo avisa."""
    ruta = os.path.join(ckpt_dir, 'config.jsonl')
    cfg = _plano(cfg)
    previo = _ultimo(ruta)
    cambios = diferencias(previo['config'], cfg) if previo else {}
    reg = dict(fecha=time.strftime('%Y-%m-%d %H:%M:%S'), version=version, commit=commit,
               argv=list(sys.argv[1:] if argv is None else argv), reanudacion=previo is not None,
               overrides=diferencias(defecto, cfg), cambios_vs_anterior=cambios, config=cfg)
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(ruta, 'a', encoding='utf-8') as f:
            f.write(json.dumps(reg, ensure_ascii=False) + '\n')
    except OSError as e:
        print(f"[config] no se pudo escribir {ruta}: {e}")
    sospechosos = {k: v for k, v in cambios.items() if k not in ESPERABLES}
    if sospechosos:
        print("[config] AVISO: esta reanudación cambia la configuración respecto al lanzamiento anterior:")
        for k, (antes, ahora) in sospechosos.items():
            print(f"[config]   {k}: {antes} -> {ahora}")
    return ruta, cambios
