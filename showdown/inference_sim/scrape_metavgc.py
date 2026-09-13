"""Scrape MetaVGC Reg M-C teams into TeamMeta JSON."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Set

UA = {'User-Agent': 'Mozilla/5.0 (compatible; IAPSVGC-inference-sim/1.0)'}
OUT = Path(__file__).resolve().parent / 'metavgc_regmc_teams.json'
BASE = 'https://metavgc.com'

SPECIES_MAP = {
    'floette': 'Floette-Eternal',
    'floette mega': 'Floette-Eternal',
    'basculegion male': 'Basculegion',
    'basculegion-male': 'Basculegion',
    'basculegion-m': 'Basculegion',
    'basculegionm': 'Basculegion',
    'basculegion female': 'Basculegion-F',
    'basculegion-female': 'Basculegion-F',
    'meowstic female': 'Meowstic-F',
    'meowstic-female': 'Meowstic-F',
    'meowstic male': 'Meowstic',
    'indeedee female': 'Indeedee-F',
    'indeedee-female': 'Indeedee-F',
    'indeedee male': 'Indeedee',
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8', 'replace')


def parse_sp(text: str) -> Dict[str, int]:
    mapping = {
        'hp': 'hp', 'atk': 'atk', 'def': 'def', 'spa': 'spa', 'spd': 'spd', 'spe': 'spe',
    }
    out = {k: 0 for k in mapping}
    for num, name in re.findall(r'(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)', text, flags=re.I):
        out[mapping[name.lower()]] = int(num)
    return out


def clean_species(name: str) -> str:
    n = unescape(name or '').strip()
    n = re.sub(r'^Mega\s+', '', n, flags=re.I)
    n = re.sub(r'\s+Mega(?:-[XY])?$', '', n, flags=re.I)
    n = re.sub(r'\s+Mega\s+[XY]$', '', n, flags=re.I)
    n = n.replace('♂', '').replace('♀', '').strip()
    key = re.sub(r'\s+', ' ', n).lower()
    if key in SPECIES_MAP:
        return SPECIES_MAP[key]
    # compact key (Basculegion-M → basculegion-m / basculegionm)
    compact = re.sub(r'[^a-z0-9]+', '', key)
    dashed = key.replace(' ', '-')
    for cand in (dashed, compact, key):
        if cand in SPECIES_MAP:
            return SPECIES_MAP[cand]
    return n


def extract_mons_from_html(html: str) -> List[dict]:
    blocks = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>', html, flags=re.S
    )
    members = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        ent = data.get('mainEntity')
        if isinstance(ent, dict) and ent.get('@type') == 'SportsTeam':
            members = ent.get('member') or []
            if members:
                break
    mons: List[dict] = []
    for thing in members:
        if not isinstance(thing, dict):
            continue
        desc = thing.get('description') or ''
        if 'Ability:' not in desc or 'Item:' not in desc:
            continue
        name = clean_species(thing.get('name') or '')
        ability = re.search(r'Ability:\s*([^|]+)', desc)
        item = re.search(r'Item:\s*([^|]+)', desc)
        nature = re.search(r'Nature:\s*([^|]+)', desc)
        moves_m = re.search(r'Moves:\s*(.+)$', desc)
        moves = [m.strip() for m in (moves_m.group(1).split(',') if moves_m else []) if m.strip()]
        # Algunas fichas MetaVGC listan <4 moves en el JSON-LD (p.ej. Archaludon).
        if not name or not item or len(moves) < 1:
            continue
        mons.append({
            'species': name,
            'ability': (ability.group(1).strip() if ability else ''),
            'item': (item.group(1).strip() if item else ''),
            'nature': (nature.group(1).strip() if nature else 'Serious'),
            'moves': moves[:4],
        })
    return mons[:6]


def extract_evs(html: str) -> List[Dict[str, int]]:
    titles = re.findall(r'title="EVs:\s*([^"]+)"', html)
    return [parse_sp(t) for t in titles[:6]]


def extract_title(html: str, slug: str) -> str:
    m = re.search(r'<title>(.*?)</title>', html, flags=re.S | re.I)
    if m:
        t = unescape(re.sub(r'\s+', ' ', m.group(1))).strip()
        if '—' in t:
            t = t.split('—', 1)[-1].strip()
        return t[:140]
    return slug


def parse_team_page(url: str, html: str) -> Optional[dict]:
    slug = url.rstrip('/').split('/')[-1]
    mons = extract_mons_from_html(html)
    evs = extract_evs(html)
    if len(mons) < 6 or len(evs) < 6:
        return None
    out_mons = []
    for mon, sp in zip(mons, evs):
        entry = {**mon, 'sp': sp}
        if entry['species'] == 'Basculegion':
            entry['gender'] = 'M'
        elif entry['species'] == 'Basculegion-F':
            entry['gender'] = 'F'
        out_mons.append(entry)
    return {
        'name': extract_title(html, slug),
        'source_url': url,
        'slug': slug,
        'mons': out_mons,
    }


def team_links_in(html: str) -> List[str]:
    out = []
    for path in re.findall(r'href="(/teams/[^"#?]+)"', html):
        if path in ('/teams/finder', '/teams/tournaments', '/teams'):
            continue
        if '/featured/' in path:
            continue
        if path.count('/') < 2:
            continue
        out.append(BASE + path)
    return out


def is_reg_mc_slug(slug: str) -> bool:
    s = slug.lower()
    return any(
        x in s
        for x in (
            'regulation-m-c',
            'reg-m-c',
            'reg-mc',
            'regulation-mc',
            'champions-reg-m-c',
            'champions-reg',
        )
    )


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


def _load_existing() -> List[dict]:
    if not OUT.is_file():
        return []
    try:
        raw = json.loads(OUT.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return []
    return raw if isinstance(raw, list) else []


def _seed_urls() -> List[str]:
    pokemon = [
        'salamence-mega', 'floette-mega', 'rillaboom', 'sneasler', 'gholdengo',
        'incineroar', 'raichu-mega-y', 'golisopod-mega', 'swampert-mega',
        'kingambit', 'archaludon', 'farigiraf', 'whimsicott', 'basculegion-male',
        'charizard-mega-y', 'gengar-mega', 'metagross-mega', 'lucario-mega',
        'tyranitar-mega', 'pelipper', 'indeedee-female', 'ursaluna', 'sinistcha',
        'arcanine-hisui', 'blaziken', 'meganium-mega', 'snorlax', 'grimmsnarl',
        'amoonguss', 'porygon2', 'heatran', 'landorus-therian', 'flutter-mane',
        'chien-pao', 'iron-hands', 'ogerpon-wellspring', 'urshifu-rapid-strike',
    ]
    seeds = [
        f'{BASE}/teams/featured/regulation-mc',
        f'{BASE}/teams/finder?format=pokemonchampions-regc',
    ]
    seeds.extend(f'{BASE}/pokemon/{slug}' for slug in pokemon)
    featured_path = Path(__file__).resolve().parent / '_featured_urls.json'
    if featured_path.is_file():
        try:
            paths = json.loads(featured_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            paths = []
        for path in paths:
            if isinstance(path, str) and path.startswith('/teams/'):
                seeds.append(BASE + path)
    return seeds


def collect_and_parse(
    want: int = 20,
    crawl_cap: int = 80,
    *,
    skip_slugs: Optional[Set[str]] = None,
) -> List[dict]:
    skip = set(skip_slugs or ())
    seeds = _seed_urls()
    teams: List[dict] = []
    errors: List[tuple] = []
    seen: Set[str] = set()
    queue: List[str] = []
    sigs: Set[str] = set(skip)
    # Featured list / featured page: aceptar aunque el slug no diga "reg-m-c".
    trusted: Set[str] = set()

    for seed in seeds:
        if '/teams/' in seed and '/featured/' not in seed and 'finder' not in seed:
            slug = seed.rstrip('/').split('/')[-1]
            if seed in seen or slug in skip:
                continue
            seen.add(seed)
            trusted.add(seed)
            queue.append(seed)
            continue
        try:
            html = fetch(seed)
        except Exception as exc:
            _safe_print(f'seed fail {seed} {exc}')
            continue
        for full in team_links_in(html):
            slug = full.rstrip('/').split('/')[-1]
            if full in seen or slug in skip:
                continue
            if '/teams/featured/' in seed:
                trusted.add(full)
            elif not is_reg_mc_slug(slug):
                continue
            seen.add(full)
            queue.append(full)

    visited = 0
    while queue and len(teams) < want and visited < crawl_cap:
        url = queue.pop(0)
        visited += 1
        slug = url.rstrip('/').split('/')[-1]
        if slug in skip or slug in sigs:
            continue
        if url not in trusted and not is_reg_mc_slug(slug):
            continue
        try:
            html = fetch(url)
        except Exception as exc:
            errors.append((url, repr(exc)))
            continue
        for full in team_links_in(html):
            link_slug = full.rstrip('/').split('/')[-1]
            if full in seen or link_slug in skip:
                continue
            if not is_reg_mc_slug(link_slug):
                continue
            seen.add(full)
            queue.append(full)

        team = parse_team_page(url, html)
        if not team:
            errors.append((url, 'parse incomplete'))
            _safe_print(f'SKIP incomplete {slug[:60]}')
            time.sleep(0.12)
            continue
        name_l = (team.get('name') or '').lower()
        if any(bad in name_l for bad in ('reg m-b', 'reg-mb', 'regulation m-b', 'reg m-a')):
            errors.append((url, 'wrong format'))
            _safe_print(f'SKIP wrong format {slug[:50]}')
            time.sleep(0.12)
            continue
        sig = team.get('slug') or team.get('source_url') or ''
        if sig in sigs:
            _safe_print(f'SKIP duplicate url {(team.get("name") or "")[:50]}')
            time.sleep(0.12)
            continue
        sigs.add(sig)
        teams.append(team)
        _safe_print(f'OK [{len(teams)}/{want}] {team["name"]}')
        time.sleep(0.12)

    _safe_print(f'visited={visited} queued_left={len(queue)} errors={len(errors)}')
    for u, e in errors[:12]:
        _safe_print(f' err {e} {u.split("/")[-1][:50]}')
    return teams


def append_new_teams(want_new: int = 100, crawl_cap: int = 400) -> List[dict]:
    """Descarga hasta want_new equipos nuevos y los fusiona en OUT."""
    existing = _load_existing()
    have = {t.get('slug') for t in existing if t.get('slug')}
    _safe_print(f'existing={len(existing)} want_new={want_new}')
    added = collect_and_parse(want_new, crawl_cap=crawl_cap, skip_slugs=have)
    merged = existing + added
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
    _safe_print(f'Done added={len(added)} total={len(merged)} -> {OUT}')
    return added


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Scrape MetaVGC Reg M-C teams')
    parser.add_argument(
        '--append', type=int, default=0,
        help='Descargar N equipos nuevos y anexarlos al JSON existente',
    )
    parser.add_argument('--want', type=int, default=20, help='Reemplazo completo (sin --append)')
    parser.add_argument('--crawl-cap', type=int, default=400)
    args = parser.parse_args()
    if args.append > 0:
        append_new_teams(args.append, crawl_cap=args.crawl_cap)
        return
    teams = collect_and_parse(args.want, crawl_cap=args.crawl_cap)
    OUT.write_text(json.dumps(teams, indent=2, ensure_ascii=False), encoding='utf-8')
    _safe_print(f'Wrote {len(teams)} teams -> {OUT}')


if __name__ == '__main__':
    main()
