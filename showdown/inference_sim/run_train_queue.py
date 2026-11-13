"""Cola de entrenamiento/experimentación secuencial para el servidor.

Ejecuta los pasos de un archivo JSON **uno detrás de otro** (nunca dos métodos
distintos a la vez): cada paso dispone de todo el servidor y paraleliza por
dentro (``--num-envs``, ``--procs``, ``--parallel-trials``).

- Log por paso: ``<run_dir>/<nombre>.log`` (salida completa, también en consola).
- Reanudación: un paso con ``<run_dir>/<nombre>.done`` se salta; si la cola se
  corta (reinicio, Ctrl+C) basta relanzar el mismo comando.
- Marcadores en los argumentos, resueltos según el hardware detectado::

    {run_dir}   carpeta de la corrida (fuera del repo por defecto)
    {num_envs}  entornos paralelos recomendados (núcleos - 4, tope 28)
    {procs}     procesos para robust E2E (igual que num_envs)
    {gpus}      GPUs visibles (para --parallel-trials)
    {device}    cuda si hay GPU, si no cpu
    {python}    intérprete actual

Formato del archivo::

    {
      "name": "tesis_server",
      "steps": [
        {"name": "ddqn_combate", "module": "showdown.rl.ddqn.train",
         "args": ["--seed", "0", "--num-envs", "{num_envs}", "--device", "{device}",
                  "--checkpoint-dir", "{run_dir}/ddqn"]}
      ]
    }

Ejemplos::

  python -m showdown.inference_sim.run_train_queue configs/server_queue.example.json --dry-run
  python -m showdown.inference_sim.run_train_queue configs/server_queue.example.json
  python -m showdown.inference_sim.run_train_queue cola.json --only ppo_combate --force
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT


@dataclass
class QueueStep:
    name: str
    module: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout_h: Optional[float] = None
    allow_fail: bool = False


@dataclass
class TrainQueue:
    name: str
    steps: List[QueueStep]


def load_queue(path: Path) -> TrainQueue:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    steps = []
    seen = set()
    for raw in data.get('steps') or []:
        st = QueueStep(
            name=str(raw['name']),
            module=str(raw['module']),
            args=[str(a) for a in raw.get('args') or []],
            env={str(k): str(v) for k, v in (raw.get('env') or {}).items()},
            timeout_h=float(raw['timeout_h']) if raw.get('timeout_h') else None,
            allow_fail=bool(raw.get('allow_fail', False)),
        )
        if st.name in seen:
            raise ValueError(f'paso duplicado en la cola: {st.name!r}')
        seen.add(st.name)
        steps.append(st)
    if not steps:
        raise ValueError(f'la cola {path} no tiene pasos')
    return TrainQueue(name=str(data.get('name') or Path(path).stem), steps=steps)


def hardware_placeholders(run_dir: Path) -> Dict[str, str]:
    from showdown.rl.parallel import default_num_envs, gpu_count, resolve_device

    n = default_num_envs()
    return {
        'run_dir': str(run_dir),
        'num_envs': str(n),
        'procs': str(n),
        'gpus': str(max(1, gpu_count())),
        'device': resolve_device('auto'),
        'python': sys.executable,
    }


def resolve_args(args: Sequence[str], values: Dict[str, str]) -> List[str]:
    out = []
    for a in args:
        try:
            out.append(str(a).format(**values))
        except KeyError as exc:
            raise ValueError(f'marcador desconocido {exc} en argumento {a!r}') from exc
    return out


def _utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _run_step(step: QueueStep, cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    """Ejecuta el paso volcando la salida a consola y a su log."""
    timeout_s = None if step.timeout_h is None else float(step.timeout_h) * 3600.0
    t0 = time.monotonic()
    with log_path.open('a', encoding='utf-8', errors='replace') as log:
        log.write(f'\n===== {_utc()} START {step.name}\n$ {" ".join(cmd)}\n')
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                if timeout_s is not None and time.monotonic() - t0 > timeout_s:
                    proc.kill()
                    log.write(f'\n[queue] {step.name}: timeout {step.timeout_h}h, proceso terminado\n')
                    break
        except KeyboardInterrupt:
            proc.terminate()
            raise
        rc = proc.wait()
        log.write(f'===== {_utc()} END {step.name} rc={rc} '
                  f'elapsed={time.monotonic() - t0:.0f}s\n')
    return int(rc)


def run_queue(
    queue: TrainQueue,
    *,
    run_dir: Path,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    from showdown.rl.parallel import describe_resources, raise_nofile_limit

    raise_nofile_limit()
    run_dir.mkdir(parents=True, exist_ok=True)
    values = hardware_placeholders(run_dir)
    selected = [s for s in queue.steps if not only or s.name in set(only)]
    if only:
        missing = sorted(set(only) - {s.name for s in queue.steps})
        if missing:
            raise ValueError(f'pasos inexistentes: {missing}')
    print(
        f'[queue] {queue.name}: {len(selected)} paso(s) | run_dir={run_dir} | '
        f'{describe_resources()} | num_envs={values["num_envs"]} device={values["device"]}',
        flush=True,
    )
    summary = []
    for k, step in enumerate(selected, start=1):
        done_mark = run_dir / f'{step.name}.done'
        cmd = [sys.executable, '-m', step.module, *resolve_args(step.args, values)]
        if done_mark.exists() and not force:
            print(f'[queue] ({k}/{len(selected)}) {step.name}: ya completado, se salta', flush=True)
            summary.append((step.name, 'skip'))
            continue
        print(f'[queue] ({k}/{len(selected)}) {step.name}: {" ".join(cmd)}', flush=True)
        if dry_run:
            summary.append((step.name, 'dry-run'))
            continue
        env = dict(os.environ)
        env.update({kk: v.format(**values) for kk, v in step.env.items()})
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        env.setdefault('PYTHONUNBUFFERED', '1')
        rc = _run_step(step, cmd, run_dir / f'{step.name}.log', env)
        if rc == 0:
            done_mark.write_text(_utc() + '\n', encoding='utf-8')
            summary.append((step.name, 'ok'))
        elif step.allow_fail:
            summary.append((step.name, f'fail rc={rc} (permitido)'))
        else:
            summary.append((step.name, f'fail rc={rc}'))
            print(f'[queue] {step.name} falló (rc={rc}); la cola se detiene. '
                  f'Log: {run_dir / (step.name + ".log")}', flush=True)
            break
    print('[queue] resumen: ' + ', '.join(f'{n}={st}' for n, st in summary), flush=True)
    return 0 if all(not st.startswith('fail rc') or 'permitido' in st for _n, st in summary) else 1


def default_run_dir(queue_name: str) -> Path:
    """Fuera del repo: así ``git pull`` en el servidor nunca choca con resultados."""
    base = os.environ.get('IAPSVGC_RUNS_DIR') or str(Path.home() / 'iapsvgc_runs')
    return Path(base) / queue_name


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Cola secuencial de entrenamientos (un método a la vez)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('queue', help='Archivo JSON con los pasos')
    ap.add_argument('--run-dir', default=None,
                    help='Carpeta de resultados (default ~/iapsvgc_runs/<nombre>)')
    ap.add_argument('--only', nargs='*', default=None, help='Ejecutar solo estos pasos')
    ap.add_argument('--force', action='store_true', help='Repetir pasos ya completados')
    ap.add_argument('--dry-run', action='store_true', help='Mostrar comandos sin ejecutar')
    args = ap.parse_args(list(argv) if argv is not None else None)

    queue = load_queue(Path(args.queue))
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(queue.name)
    return run_queue(queue, run_dir=run_dir, only=args.only, force=args.force,
                     dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
