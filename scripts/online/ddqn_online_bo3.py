"""Partida online contra humanos: DDQN v1 reta a un usuario a un BO3 en Showdown (20-11-2026).

Igual que ``ppo_online_bo3.py`` (PPO v1), pero con DDQN v1 = kick-off oficial de DDQN (ddqn_spec 1.24,
memorizacion_equipos §18.7): observación v2.1 (P5, **calculadora real**) y preview v2.1 sin `val`.
Corre en el laptop (CPU) con el mismo jugador de poke-env que la evaluación
(``showdown.rl.ddqn.player.DDQNPlayer``, sin espejo IAP) y el equipo fijo
(``showdown/rl/data/ddqn_team.txt``).

- **Combate:** determinista (argmax de Q, épsilon 0), como en la evaluación.
- **Preview:** no determinista contra personas (decisión del usuario, P1b): sorteo con temperatura 1,0
  entre las 90 formaciones canónicas. ``--preview-greedy`` vuelve al modo determinista de la evaluación.
- **Calculadora real:** la red se entrenó con daños de ``@smogon/calc`` (``../damage-calc``, bacbbfc).
  Antes de conectarse se comprueba que responde; si no, se aborta (con la aproximación de respaldo la
  observación sería distinta a la del entrenamiento). Al final se reportan respaldos y pedidos sin
  respuesta.

Credenciales: ``PS_USERNAME`` / ``PS_PASSWORD`` del ``.env`` (no se imprimen).

    # Comprobación sin conectarse (checkpoint, preview, calculadora):
    python scripts/online/ddqn_online_bo3.py --check
    # Reto BO3:
    python scripts/online/ddqn_online_bo3.py --opponent velascoAlvarado
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_online_bo3 import _read_env  # noqa: E402

OUT = REPO / 'logs' / 'online_ddqn'


def _calc_preflight() -> str:
    """Un cálculo real (Rillaboom Wood Hammer contra Primarina, equipo fijo); lanza si no responde."""
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke, build_calc_request
    from showdown.damage_calc_bridge import damage_ranges_batch
    req = build_calc_request(CalcPoke(species='Rillaboom', ability='Grassy Surge'),
                             CalcPoke(species='Primarina', ability='Liquid Voice'), 'Wood Hammer')
    t0 = time.time()
    out = damage_ranges_batch([req])
    if not out or out[0] is None:
        raise RuntimeError('la calculadora real no respondió (¿node y ../damage-calc compilado?)')
    return f'calculadora real OK: {out[0]} en {1000 * (time.time() - t0):.0f} ms'


def _setup(args):
    """Carga y verifica todo lo que no necesita conexión. Devuelve (ck, pv, team) o lanza."""
    from showdown.inference_sim.team_factory import load_team_file, to_showdown_export
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    from showdown.rl.frozen_preview import configure_preview

    ck = Path(args.checkpoint).resolve()
    pv = Path(args.preview).resolve()
    for p in (ck, pv):
        if not p.is_file():
            raise FileNotFoundError(f'falta {p} (copiarlo del servidor; ver docs/ddqn_kickoff_checklist.md)')
    agent = DDQNAgent(DDQNConfig(device='cpu'))
    agent.load(str(ck))
    if not agent.cfg.obs_abilities:
        raise RuntimeError(f'{ck.name} no usa la observación v2.1 (obs_abilities=False): no es DDQN v1')
    print(f'[online] checkpoint {ck.name}: obs_abilities=True', flush=True)

    if args.preview_greedy:
        os.environ.pop('PREVIEW_SAMPLE', None)
        os.environ.pop('PREVIEW_TEMPERATURE', None)
    else:
        os.environ['PREVIEW_SAMPLE'] = '1'
        os.environ['PREVIEW_TEMPERATURE'] = str(args.preview_temperature)
    # Preview de la familia (team_preview_ppo), obligatorio (sin caer a /team 1234).
    print('[online] ' + configure_preview('ddqn', checkpoint=str(pv), required=True), flush=True)
    print('[online] preview ' + ('determinista' if args.preview_greedy
                                 else f'con sorteo, temperatura {args.preview_temperature}'), flush=True)
    if os.environ.get('RL_OBS_CALC', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        raise RuntimeError('RL_OBS_CALC=0: DDQN v1 necesita la calculadora real')
    print('[online] ' + _calc_preflight(), flush=True)
    team = to_showdown_export(load_team_file(REPO / args.team))
    return ck, pv, team


async def main_async(args) -> int:
    from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration
    from showdown.rl.ddqn.player import DDQNPlayer
    from showdown.rl.observation import calc_stats

    try:
        ck, pv, team = _setup(args)
    except Exception as exc:
        print(f'[online] ERROR: {exc}', flush=True)
        return 2
    if args.check:
        print('[online] comprobación completa; listo para jugar (sin --check)', flush=True)
        return 0

    env = _read_env(REPO / '.env')
    user = os.environ.get('PS_USERNAME') or env.get('PS_USERNAME')
    pwd = os.environ.get('PS_PASSWORD') or env.get('PS_PASSWORD')
    if not user or not pwd:
        print('[online] faltan PS_USERNAME / PS_PASSWORD en .env', flush=True)
        return 2
    (OUT / 'replays').mkdir(parents=True, exist_ok=True)

    player = DDQNPlayer(
        account_configuration=AccountConfiguration(user, pwd),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
        team=team,
        checkpoint=str(ck),
        preview_checkpoint=str(pv),
        epsilon=0.0,
        required=True,
        save_replays=str(OUT / 'replays'),
        start_timer_on_battle_start=False,
        log_level=logging.INFO,
    )
    print(f'[online] {user} reta a {args.opponent} ({args.format}); DDQN v1 checkpoint={ck.name}', flush=True)
    t0 = time.time()
    await player.send_challenges(args.opponent, n_challenges=1)

    games = []
    for tag, b in player.battles.items():
        games.append({'sala': tag, 'turnos': b.turn, 'ganó': b.won,
                      'rival': b.opponent_username,
                      'equipo_rival': sorted(m.species for m in b.opponent_team.values())})
    wins = sum(1 for g in games if g['ganó'])
    losses = sum(1 for g in games if g['ganó'] is False)
    cs = calc_stats()
    summary = {'agente': 'DDQN v1', 'rival': args.opponent, 'formato': args.format,
               'checkpoint': str(ck), 'preview': str(pv),
               'preview_modo': 'determinista' if args.preview_greedy else f'sorteo T={args.preview_temperature}',
               'partidas': games, 'victorias': wins, 'derrotas': losses,
               'serie': 'ganada' if wins > losses else 'perdida',
               'decisiones': player.decision_stats.as_dict(),
               'calculadora': cs,
               'duracion_min': round((time.time() - t0) / 60, 1)}
    stamp = time.strftime('%Y%m%d_%H%M%S')
    (OUT / f'bo3_{args.opponent}_{stamp}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if cs.get('fallback_obs', 0) or cs.get('missing', 0):
        print('[online] AVISO: hubo observaciones con la aproximación de respaldo o pedidos sin respuesta '
              '(la entrada no fue idéntica a la del entrenamiento en esos turnos)', flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--opponent', help='usuario de Showdown a retar (obligatorio salvo con --check)')
    ap.add_argument('--checkpoint', default=str(OUT / 'ddqn_v1_s0_final.pt'))
    ap.add_argument('--preview', default=str(OUT / 'preview_v21b_s0.pt'))
    ap.add_argument('--team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--format', default='gen9championsvgc2026regmcbo3')
    ap.add_argument('--preview-temperature', type=float, default=1.0)
    ap.add_argument('--preview-greedy', action='store_true',
                    help='preview determinista (como en la evaluación) en vez de sorteo')
    ap.add_argument('--check', action='store_true', help='solo verifica checkpoint, preview y calculadora')
    args = ap.parse_args(argv)
    if not args.check and not args.opponent:
        ap.error('--opponent es obligatorio (o usa --check)')
    os.environ['DDQN_DEVICE'] = 'cpu'
    os.environ['PREVIEW_DEVICE'] = 'cpu'
    os.environ.pop('PS_RL_STRICT', None)  # un error en un turno no debe tumbar la serie (se cuenta)
    return asyncio.run(main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
