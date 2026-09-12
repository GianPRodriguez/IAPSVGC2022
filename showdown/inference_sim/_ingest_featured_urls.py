"""Parse candidate URLs from featured pagination and append N new teams."""
from __future__ import annotations

import json
import time
from pathlib import Path

from scrape_metavgc import BASE, OUT, fetch, parse_team_page

CAND = Path(__file__).resolve().parent / '_featured_urls.json'


def main(want_new: int = 30) -> None:
    paths = json.loads(CAND.read_text(encoding='utf-8'))
    existing = json.loads(OUT.read_text(encoding='utf-8')) if OUT.is_file() else []
    have = {t.get('slug') for t in existing if t.get('slug')}
    print(f'existing={len(existing)} candidates={len(paths)} want_new={want_new}')

    added = []
    errors = 0
    for path in paths:
        if len(added) >= want_new:
            break
        slug = path.rstrip('/').split('/')[-1]
        if slug in have:
            continue
        url = BASE + path if path.startswith('/') else path
        try:
            html = fetch(url)
            team = parse_team_page(url, html)
        except Exception as exc:
            errors += 1
            print('ERR', slug[:55], exc)
            time.sleep(0.1)
            continue
        if not team:
            errors += 1
            print('SKIP', slug[:55])
            time.sleep(0.1)
            continue
        if team['slug'] in have:
            continue
        have.add(team['slug'])
        added.append(team)
        print(f'OK [{len(added)}/{want_new}] {team["name"][:70].encode("ascii", "replace").decode("ascii")}')
        time.sleep(0.12)

    merged = existing + added
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Done added={len(added)} errors={errors} total={len(merged)} -> {OUT}')


if __name__ == '__main__':
    main(30)
