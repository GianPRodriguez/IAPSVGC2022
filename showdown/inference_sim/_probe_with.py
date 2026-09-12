"""Deep probe: /teams/with/* and expand from existing team pages."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from scrape_metavgc import BASE, OUT, fetch, is_reg_mc_slug, team_links_in

LOG = Path(__file__).resolve().parents[2] / 'logs' / '_metavgc_deep.txt'


def main() -> None:
    lines = []
    with_urls = [
        f'{BASE}/teams/with/salamence-mega',
        f'{BASE}/teams/with/rillaboom',
        f'{BASE}/teams/with/floette-mega',
        f'{BASE}/teams/with/sneasler',
        f'{BASE}/teams/with/golisopod-mega',
        f'{BASE}/teams/with/swampert-mega',
        f'{BASE}/teams/with/kingambit',
        f'{BASE}/teams/with/incineroar',
        f'{BASE}/teams/with/gholdengo',
        f'{BASE}/teams/with/raichu-mega-y',
        f'{BASE}/teams/with/farigiraf',
        f'{BASE}/teams/with/basculegion-male',
        f'{BASE}/teams/with/arcanine-hisui',
        f'{BASE}/teams/with/whimsicott',
        f'{BASE}/teams/with/dragonite',
        f'{BASE}/teams/with/metagross-mega',
        f'{BASE}/teams/with/charizard-mega-x',
        f'{BASE}/teams/with/blaziken',
        f'{BASE}/teams/with/sinistcha',
        f'{BASE}/teams/with/archaludon',
    ]
    existing = json.loads(OUT.read_text(encoding='utf-8')) if OUT.is_file() else []
    have = {t.get('slug') for t in existing if t.get('slug')}

    found = set()
    for u in with_urls:
        try:
            html = fetch(u)
        except Exception as exc:
            lines.append(f'FAIL {u} {exc!r}')
            continue
        links = team_links_in(html)
        # also raw hrefs that look like team pages
        raw = re.findall(r'href="(/teams/[^"#?]+)"', html)
        teamish = []
        for p in raw:
            if p.startswith('/teams/with/') or p.startswith('/teams/featured/') or p in (
                '/teams/finder', '/teams/tournaments', '/teams'
            ):
                continue
            if p.count('/') < 2:
                continue
            teamish.append(BASE + p)
        reg = [l for l in teamish if is_reg_mc_slug(l.rstrip('/').split('/')[-1])]
        new = [l for l in reg if l.rstrip('/').split('/')[-1] not in have]
        for l in new:
            found.add(l)
        lines.append(
            f'{u.split("/")[-1]}: html={len(html)} team_links={len(links)} '
            f'teamish={len(teamish)} reg={len(reg)} new={len(new)}'
        )
        time.sleep(0.1)

    lines.append(f'\nunique new from /with/: {len(found)}')
    for l in sorted(found)[:40]:
        lines.append('  ' + l.split('/')[-1][:80])

    # Also crawl neighbor links from existing team pages
    neigh = set()
    for t in existing[:8]:
        url = t.get('source_url') or ''
        if not url:
            continue
        try:
            html = fetch(url)
        except Exception:
            continue
        for full in team_links_in(html):
            slug = full.rstrip('/').split('/')[-1]
            if slug not in have and is_reg_mc_slug(slug):
                neigh.add(full)
                found.add(full)
        time.sleep(0.1)
    lines.append(f'neighbors from first 8 existing: {len(neigh)}')
    lines.append(f'total unique new candidates: {len(found)}')

    LOG.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    # write candidates for expand
    cand_path = Path(__file__).resolve().parent / '_candidates.json'
    cand_path.write_text(json.dumps(sorted(found), indent=2), encoding='utf-8')
    print(f'Wrote {cand_path} ({len(found)})')


if __name__ == '__main__':
    main()
