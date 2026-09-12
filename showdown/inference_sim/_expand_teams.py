"""Añade N equipos Reg M-C nuevos (sin duplicar slug) a metavgc_regmc_teams.json."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Set

from scrape_metavgc import (
    BASE,
    OUT,
    fetch,
    is_reg_mc_slug,
    parse_team_page,
    team_links_in,
)

# Seeds amplias: featured + finder + Pokémon populares Reg M-C
POKE_SEEDS = [
    f'{BASE}/teams/featured/regulation-mc',
    f'{BASE}/teams/finder?format=pokemonchampions-regc',
    f'{BASE}/teams/finder?format=pokemonchampions-regc&sort=recent',
    f'{BASE}/pokemon/salamence-mega',
    f'{BASE}/pokemon/floette-mega',
    f'{BASE}/pokemon/rillaboom',
    f'{BASE}/pokemon/sneasler',
    f'{BASE}/pokemon/gholdengo',
    f'{BASE}/pokemon/incineroar',
    f'{BASE}/pokemon/raichu-mega-y',
    f'{BASE}/pokemon/golisopod-mega',
    f'{BASE}/pokemon/swampert-mega',
    f'{BASE}/pokemon/kingambit',
    f'{BASE}/pokemon/farigiraf',
    f'{BASE}/pokemon/whimsicott',
    f'{BASE}/pokemon/basculegion-male',
    f'{BASE}/pokemon/sinistcha',
    f'{BASE}/pokemon/blaziken',
    f'{BASE}/pokemon/archaludon',
    f'{BASE}/pokemon/dragonite',
    f'{BASE}/pokemon/metagross-mega',
    f'{BASE}/pokemon/gengar-mega',
    f'{BASE}/pokemon/charizard-mega-y',
    f'{BASE}/pokemon/charizard-mega-x',
    f'{BASE}/pokemon/tyranitar-mega',
    f'{BASE}/pokemon/aegislash',
    f'{BASE}/pokemon/ursaluna',
    f'{BASE}/pokemon/ursaluna-bloodmoon',
    f'{BASE}/pokemon/ogerpon-hearthflame',
    f'{BASE}/pokemon/ogerpon-wellspring',
    f'{BASE}/pokemon/raging-bolt',
    f'{BASE}/pokemon/flutter-mane',
    f'{BASE}/pokemon/landorus',
    f'{BASE}/pokemon/tornadus',
    f'{BASE}/pokemon/thundurus',
    f'{BASE}/pokemon/urshifu-rapid-strike',
    f'{BASE}/pokemon/urshifu',
    f'{BASE}/pokemon/chi-yu',
    f'{BASE}/pokemon/chien-pao',
    f'{BASE}/pokemon/ting-lu',
    f'{BASE}/pokemon/wo-chien',
]


def harvest_urls(have: Set[str]) -> List[str]:
    seen: Set[str] = set()
    urls: List[str] = []
    for seed in POKE_SEEDS:
        try:
            html = fetch(seed)
        except Exception as exc:
            print('seed fail', seed.split('/')[-1][:40], exc)
            continue
        added = 0
        for full in team_links_in(html):
            slug = full.rstrip('/').split('/')[-1]
            if full in seen or slug in have:
                continue
            # En páginas de Pokémon / featured aceptamos Reg M-C por slug;
            # en finder ya vienen filtrados por format.
            if 'finder' not in seed and not is_reg_mc_slug(slug):
                continue
            seen.add(full)
            urls.append(full)
            added += 1
        print(f'seed {seed.rstrip("/").split("/")[-1][:40]} -> +{added} (pool {len(urls)})')
        time.sleep(0.15)
    return urls


def main(want_new: int = 30, crawl_cap: int = 200) -> None:
    existing: List[dict] = []
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding='utf-8'))
    have: Set[str] = {t.get('slug') for t in existing if t.get('slug')}
    print(f'existing={len(existing)} unique_slugs={len(have)} want_new={want_new}')

    urls = harvest_urls(have)
    print(f'candidate urls={len(urls)}')

    added: List[dict] = []
    visited = 0
    errors = 0
    for url in urls:
        if len(added) >= want_new:
            break
        if visited >= crawl_cap:
            break
        visited += 1
        slug = url.rstrip('/').split('/')[-1]
        if slug in have:
            continue
        try:
            html = fetch(url)
            # Enriquecer cola con links vecinos Reg M-C
            for full in team_links_in(html):
                s2 = full.rstrip('/').split('/')[-1]
                if s2 in have or full in urls:
                    continue
                if is_reg_mc_slug(s2):
                    urls.append(full)
            team = parse_team_page(url, html)
        except Exception as exc:
            errors += 1
            print('ERR', slug[:50], exc)
            time.sleep(0.12)
            continue
        if not team:
            errors += 1
            print('SKIP', slug[:55])
            time.sleep(0.12)
            continue
        if team['slug'] in have:
            print('DUP', team['slug'][:50])
            time.sleep(0.08)
            continue
        have.add(team['slug'])
        added.append(team)
        print(f'OK [{len(added)}/{want_new}] {team["name"][:70]}')
        time.sleep(0.12)

    merged = existing + added
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
    print(
        f'Done visited={visited} errors={errors} added={len(added)} '
        f'total={len(merged)} -> {OUT}'
    )


if __name__ == '__main__':
    main(30)
