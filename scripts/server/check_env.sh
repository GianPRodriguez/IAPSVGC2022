#!/usr/bin/env bash
# Verifica que el servidor esté listo para entrenar (no instala nada).
# Uso (desde la raíz del repo, con el entorno conda activo):
#   bash scripts/server/check_env.sh
set -u
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
PARENT="$(dirname "$REPO")"
ok=0; bad=0
pass() { echo "  [OK]   $*"; ok=$((ok+1)); }
fail() { echo "  [FALTA] $*"; bad=$((bad+1)); }

echo "== Repos hermanos (en $PARENT)"
for d in pokemon-showdown poke-env damage-calc; do
  [ -d "$PARENT/$d" ] && pass "$d" || fail "$d (clonar junto a IAPSVGC2022)"
done
[ -d "$PARENT/pokemon-showdown/node_modules" ] && pass "pokemon-showdown/node_modules" \
  || fail "pokemon-showdown: ejecutar 'npm install' dentro de pokemon-showdown"

echo "== Node"
command -v node >/dev/null && pass "node $(node -v)" || fail "node no está en PATH"

echo "== Python ($(command -v python))"
python - <<'PY'
import importlib, sys
print(f"  python {sys.version.split()[0]}")
mods = ["numpy", "gymnasium", "poke_env", "torch", "scipy", "matplotlib", "environs", "PIL"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m} ({exc.__class__.__name__})")
if missing:
    print("  [FALTA] módulos: " + ", ".join(missing))
else:
    print("  [OK]   módulos Python")
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"  [{'OK' if n else 'FALTA'}]   torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={n} {names}")
except Exception as exc:
    print(f"  [FALTA] torch: {exc!r}")
import poke_env, os
print(f"  poke_env desde: {os.path.dirname(poke_env.__file__)}")
from showdown.rl.parallel import describe_resources, default_num_envs, raise_nofile_limit
lim = raise_nofile_limit()
print(f"  recursos: {describe_resources()} -> num_envs recomendado={default_num_envs()} (nofile tras subir={lim})")
PY

echo "== Límites"
soft=$(ulimit -n); hard=$(ulimit -Hn)
echo "  nofile blando=$soft duro=$hard (el código sube el blando hasta 65536 solo)"

echo "== Resultado: $ok OK, $bad faltantes"
[ "$bad" -eq 0 ]
