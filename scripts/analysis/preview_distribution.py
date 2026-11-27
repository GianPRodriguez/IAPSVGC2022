"""¿Está colapsado el preview? Distribución del preview congelado sobre las formaciones (protocolo §4.16, M1).

Motivo (docs/online_web_partidas.md): en las 4 partidas contra personas el preview v2.1b trajo casi lo mismo
aun con sorteo T = 1 (Raichu 0/4, Arcanine-Hisui 4/4), contra lluvia y contra sol. Antes de pedirle que se
adapte a la serie hay que saber si reacciona al equipo rival.

Para cada equipo rival (hoja abierta: especie, objeto, habilidad y movimientos; sin SP) calcula la
distribución del modo con sorteo sobre las 90 formaciones canónicas y resume, por grupo de equipos:

- entropía media (nats, normalizada por ln 90) y número efectivo de formaciones (exp H);
- probabilidad de traer y de abrir con cada Pokémon propio (media y rango entre rivales);
- **información mutua** I(rival; formación) = H(p̄) − E[H(p)]: cuánto depende la elección del rival
  (0 = elige igual contra todos);
- distancia de variación total media entre la distribución de cada rival y la media;
- formaciones del modo determinista (argmax): cuántas distintas y las más frecuentes.

Incluye los equipos de las personas de la web (solo especies) y, para ellos, el preview condicionado a lo que
trajeron en la serie (``foe_keep``, sin reentrenar), para ver si el condicionamiento cambia algo.

    python scripts/analysis/preview_distribution.py \\
        --checkpoint logs/online_ddqn/preview_v21b_s0.pt --splits train,val,eval --out logs/preview_dist
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _entropy(p: np.ndarray) -> float:
    q = p[p > 1e-12]
    return float(-(q * np.log(q)).sum())


def summarize(dists: Sequence[np.ndarray], names: Sequence[str], n_canon: int) -> Dict[str, object]:
    """Resumen de un grupo de distribuciones (una por rival) sobre las 360 formaciones."""
    from showdown.rl.team_preview_action import decode_preview_action

    P = np.stack(dists)                      # [n, 360]
    n = P.shape[0]
    H = np.array([_entropy(p) for p in P])
    pbar = P.mean(axis=0)
    mi = _entropy(pbar) - float(H.mean())
    tv = 0.5 * np.abs(P - pbar).sum(axis=1)
    bring = np.zeros((n, 6))
    lead = np.zeros((n, 6))
    for a in np.flatnonzero(pbar > 0):
        i1, i2, i3, i4 = decode_preview_action(int(a))
        for i in (i1, i2, i3, i4):
            bring[:, i] += P[:, a]
        for i in (i1, i2):
            lead[:, i] += P[:, a]
    greedy = [decode_preview_action(int(np.argmax(p))) for p in P]
    g_bring = Counter(tuple(sorted(g)) for g in greedy)
    g_form = Counter(g for g in greedy)

    def per_mon(m: np.ndarray) -> List[Dict[str, float]]:
        return [{'pokemon': names[i], 'media': round(float(m[:, i].mean()), 4),
                 'min': round(float(m[:, i].min()), 4), 'max': round(float(m[:, i].max()), 4)}
                for i in range(6)]

    return {
        'rivales': n,
        'entropia_media_nats': round(float(H.mean()), 4),
        'entropia_normalizada': round(float(H.mean() / math.log(n_canon)), 4),
        'formaciones_efectivas_media': round(float(np.exp(H).mean()), 2),
        'informacion_mutua_nats': round(mi, 4),
        'informacion_mutua_normalizada': round(mi / math.log(n_canon), 4),
        'tv_media_a_la_media': round(float(tv.mean()), 4),
        'traer': per_mon(bring),
        'abrir': per_mon(lead),
        'determinista_traidos_distintos': len(g_bring),
        'determinista_formaciones_distintas': len(g_form),
        'determinista_traidos_top': [{'traidos': [names[i] for i in k], 'rivales': c}
                                     for k, c in g_bring.most_common(5)],
        'determinista_formaciones_top': [{'abre': [names[g[0]], names[g[1]]], 'atras': [names[g[2]], names[g[3]]],
                                          'rivales': c} for g, c in g_form.most_common(5)],
    }


def human_teams(folder: Path) -> List[Dict[str, object]]:
    """Equipos de las series de la web (solo especies) y lo que trajeron, desde los JSON de logs/online_web."""
    out = []
    for f in sorted(folder.glob('bo3_*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        games = d.get('partidas') or []
        if not games or str(d.get('rival', '')).lower() == 'deep1oddqn':  # la prueba técnica no cuenta
            continue
        team = sorted({s for g in games for s in g.get('equipo_rival', [])})
        out.append({'rival': d.get('rival'), 'archivo': f.name, 'equipo': team})
    return out


def brought_from_replays(folder: Path, rival: str) -> List[str]:
    """Especies que el rival trajo en la serie (replays de logs/online_web)."""
    import html as _html
    from showdown.rl.series_memory import GameRecord
    keep = set()
    for f in sorted((folder / 'replays').glob('*.html')):
        t = _html.unescape(f.read_text(encoding='utf-8'))
        if rival.lower() not in t.lower():
            continue
        msgs = [ln.split('|') for ln in t.split('\n') if ln.startswith('|')]
        roles = [m for m in msgs if len(m) > 3 and m[1] == 'player' and rival.lower().replace(' ', '') in
                 m[3].lower().replace(' ', '')]
        if not roles:
            continue
        our_role = 'p2' if roles[0][2] == 'p1' else 'p1'
        keep.update(GameRecord.from_messages(msgs, our_role=our_role).brought())
    return sorted(keep)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', default=str(REPO / 'logs/online_ddqn/preview_v21b_s0.pt'))
    ap.add_argument('--kind', default='team_preview_ppo')
    ap.add_argument('--agent-team', default=str(REPO / 'showdown/rl/data/ddqn_team.txt'))
    ap.add_argument('--splits', default='train,val,eval')
    ap.add_argument('--temperature', type=float, default=1.0, help='temperatura del sorteo (contra personas: 1,0)')
    ap.add_argument('--web-dir', default=str(REPO / 'logs/online_web'))
    ap.add_argument('--out', default=str(REPO / 'logs/preview_dist'))
    args = ap.parse_args(argv)

    from showdown.inference_sim.team_factory import active_teams, load_team_file
    from showdown.inference_sim.team_split import team_key
    from showdown.rl.frozen_preview import FrozenPreviewPolicy
    from showdown.rl.team_preview_action import canonical_mask
    from showdown.rl.team_preview_obs import mon_feat_from_monset, mon_feat_from_species

    pol = FrozenPreviewPolicy(kind=args.kind, checkpoint=args.checkpoint, greedy=False,
                              temperature=args.temperature)
    pol._ensure()
    if pol._agent is None:
        print(f'[preview_dist] no se pudo cargar {args.checkpoint}: {pol._load_error}', flush=True)
        return 2
    n_canon = int(sum(canonical_mask())) if getattr(pol, '_canon', None) is not None else 360
    ours = load_team_file(args.agent_team)
    our_feats = [mon_feat_from_monset(m, public_only=False) for m in ours.mons]
    names = [str(m.species) for m in ours.mons]
    our_key = team_key(ours)

    report: Dict[str, object] = {'checkpoint': args.checkpoint, 'temperatura': args.temperature,
                                 'formaciones_canonicas': n_canon, 'equipo_agente': names, 'grupos': {}}
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        teams = [t for t in active_teams(split) if team_key(t) != our_key]
        dists = [pol.distribution_from_feats(our_feats, [mon_feat_from_monset(m, public_only=True) for m in t.mons])
                 for t in teams]
        report['grupos'][split] = summarize(dists, names, n_canon)
        print(f'[preview_dist] {split}: {len(teams)} rivales', flush=True)

    humans = human_teams(Path(args.web_dir))
    if humans:
        rows = []
        for h in humans:
            foe = [mon_feat_from_species(s) for s in h['equipo']]
            full = pol.distribution_from_feats(our_feats, foe)
            kept = brought_from_replays(Path(args.web_dir), str(h['rival']))
            from showdown.rl.series_memory import species_key
            foe_k = [mon_feat_from_species(s) for s in h['equipo'] if species_key(s) in set(kept)] if kept else foe
            cond = pol.distribution_from_feats(our_feats, foe_k)
            rows.append({'rival': h['rival'], 'equipo': h['equipo'], 'trajo': kept,
                         'completo': summarize([full], names, n_canon),
                         'condicionado': summarize([cond], names, n_canon)})
        report['personas'] = rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    (out / f'preview_dist_{stem}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for split, r in report['grupos'].items():
        print(f'\n== {split}: H/ln{n_canon} = {r["entropia_normalizada"]}, formaciones efectivas = '
              f'{r["formaciones_efectivas_media"]}, I(rival;formación)/ln{n_canon} = {r["informacion_mutua_normalizada"]}, '
              f'TV media = {r["tv_media_a_la_media"]}, argmax distintos = {r["determinista_formaciones_distintas"]}')
        print('   traer: ' + ', '.join(f'{x["pokemon"]} {x["media"]:.2f} [{x["min"]:.2f}–{x["max"]:.2f}]'
                                     for x in r['traer']))
    for row in report.get('personas', []):
        print(f'\n== persona {row["rival"]} (trajo {row["trajo"]})')
        for k in ('completo', 'condicionado'):
            r = row[k]
            print(f'   {k}: ' + ', '.join(f'{x["pokemon"]} {x["media"]:.2f}' for x in r['traer'])
                  + f' | argmax {r["determinista_formaciones_top"][0]}')
    print(f'\n[preview_dist] -> {out / f"preview_dist_{stem}.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
