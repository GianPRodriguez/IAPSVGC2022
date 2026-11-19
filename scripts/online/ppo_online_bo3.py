"""Partida de prueba online: el PPO del kick-off oficial reta a un usuario a un BO3 en Showdown.

Corre en el laptop (CPU). Usa el mismo jugador de poke-env y la misma observación que el
entrenamiento y la evaluación (``showdown.rl.ppo.player.PPOPlayer``, sin espejo IAP), el preview
v2 congelado y el equipo fijo (``showdown/rl/data/ddqn_team.txt``). Política determinista (greedy).

Credenciales: ``PS_USERNAME`` / ``PS_PASSWORD`` del ``.env`` (no se imprimen).

    python scripts/online/ppo_online_bo3.py --opponent velascoAlvarado \
        --checkpoint logs/online_ppo/ppo_s2_final.pt --preview logs/online_ppo/preview_v2.pt
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


def _read_env(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


async def main_async(args) -> int:
    from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration
    from showdown.inference_sim.team_factory import load_team_file, to_showdown_export
    from showdown.rl.frozen_preview import configure_preview
    from showdown.rl.ppo.player import PPOPlayer

    env = _read_env(REPO / '.env')
    user = os.environ.get('PS_USERNAME') or env.get('PS_USERNAME')
    pwd = os.environ.get('PS_PASSWORD') or env.get('PS_PASSWORD')
    if not user or not pwd:
        print('[online] faltan PS_USERNAME / PS_PASSWORD en .env', flush=True)
        return 2
    ck = Path(args.checkpoint).resolve()
    pv = Path(args.preview).resolve()
    for p in (ck, pv):
        if not p.is_file():
            print(f'[online] falta {p}', flush=True)
            return 2

    # Igual que policy_diag: preview de la familia PPO, obligatorio (sin caer a /team 1234).
    print('[online] ' + configure_preview('ppo', checkpoint=str(pv), required=True), flush=True)
    team = to_showdown_export(load_team_file(REPO / args.team))
    out = REPO / 'logs' / 'online_ppo'
    (out / 'replays').mkdir(parents=True, exist_ok=True)

    player = PPOPlayer(
        account_configuration=AccountConfiguration(user, pwd),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
        team=team,
        checkpoint=str(ck),
        preview_checkpoint=str(pv),
        greedy=True,
        save_replays=str(out / 'replays'),
        start_timer_on_battle_start=False,
        log_level=logging.INFO,
    )
    if not player.agent or not player.checkpoint_path:
        print('[online] no se cargó el checkpoint', flush=True)
        return 2
    print(f'[online] {user} reta a {args.opponent} ({args.format}); checkpoint={ck.name}', flush=True)
    t0 = time.time()
    await player.send_challenges(args.opponent, n_challenges=1)

    games = []
    for tag, b in player.battles.items():
        games.append({'sala': tag, 'turnos': b.turn, 'ganó': b.won,
                      'rival': b.opponent_username,
                      'equipo_rival': sorted(m.species for m in b.opponent_team.values())})
    wins = sum(1 for g in games if g['ganó'])
    losses = sum(1 for g in games if g['ganó'] is False)
    summary = {'rival': args.opponent, 'formato': args.format, 'checkpoint': str(ck),
               'preview': str(pv), 'partidas': games, 'victorias': wins, 'derrotas': losses,
               'serie': 'ganada' if wins > losses else 'perdida',
               'decisiones': player.decision_stats.as_dict(),
               'duracion_min': round((time.time() - t0) / 60, 1)}
    stamp = time.strftime('%Y%m%d_%H%M%S')
    (out / f'bo3_{args.opponent}_{stamp}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--opponent', required=True)
    ap.add_argument('--checkpoint', default='logs/online_ppo/ppo_s2_final.pt')
    ap.add_argument('--preview', default='logs/online_ppo/preview_v2.pt')
    ap.add_argument('--team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--format', default='gen9championsvgc2026regmcbo3')
    args = ap.parse_args(argv)
    os.environ.setdefault('PPO_DEVICE', 'cpu')
    os.environ.pop('PS_RL_STRICT', None)  # un error en un turno no debe tumbar la serie (se cuenta)
    return asyncio.run(main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
