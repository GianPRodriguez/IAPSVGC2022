"""Corre el archivo de pruebas sin pytest (no esta instalado en este entorno)."""
import sys
import traceback

sys.path.insert(0, '.')

import tests.test_reserve_and_ditto_regression as t

names = sorted(n for n in dir(t) if n.startswith('test_'))
fails = []
for name in names:
    try:
        getattr(t, name)()
    except Exception:
        fails.append((name, traceback.format_exc()))

for name, tb in fails:
    print('=' * 70)
    print('FAIL', name)
    print(tb)

print(f'PASS {len(names) - len(fails)} / {len(names)}')
